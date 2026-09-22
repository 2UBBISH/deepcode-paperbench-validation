"""Masked denoising score-matching training for Simformer.

Implements the training objective of Sec. 3.3 of the Simformer paper,

    x_hat_t^{M_C} = (1 - M_C) * x_hat_t + M_C * x_hat_0          (conditioned vars stay clean)
    l(phi, M_C, t, x_hat_0, x_hat_t) =
        (1 - M_C) * ( s_phi^{M_E}(x_hat_t^{M_C}, t) - grad_{x_hat_t} log p_t(x_hat_t | x_hat_0) )
    L(phi) = E_{M_C, t, x_hat_0, x_hat_t}[ || l(phi, M_C, t, x_hat_0, x_hat_t) ||_2^2 ]

together with the optimisation recipe of Appendix A2.1 (batch size 1000, Adam, early stopping
on validation loss, uniform time sampling, condition masks drawn uniformly from
{joint, posterior, likelihood, rand(Bernoulli 0.3), rand(Bernoulli 0.7)}).

The score network of this codebase predicts epsilon (i.e. the unit-variance noise), which is
related to the score by ``score = -eps / sigma(t)``; the loss can therefore be evaluated either
in score space (default, exactly Eq. 1 above) or in epsilon space (identical up to the
per-element weighting ``sigma(t)^-2``).

The module is PyTorch based (matching ``tokenizer.py``/``transformer.py``); the SDE coefficients
come from ``diffusion.py`` and are evaluated through their (framework agnostic) marginal
mean/std.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:  # torch is a hard requirement for the neural part of the code base
    import torch

    _HAS_TORCH = True
    _TORCH_IMPORT_ERROR: Optional[BaseException] = None
except Exception as _exc:  # pragma: no cover - only hit in a torch-less environment
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False
    _TORCH_IMPORT_ERROR = _exc

from .condition_masks import (
    DEFAULT_MODES,
    P_RANDOM_HIGH,
    P_RANDOM_LOW,
    ConditionMaskSampler,
    group_broadcast,
)
from .diffusion import SDE, SDEConfig, get_sde, sde_from_config

__all__ = [
    # config
    "TrainingConfig",
    "Standardizer",
    # mask / noise helpers
    "variable_expansion_widths",
    "expand_variable_mask",
    "collapse_value_mask",
    "token_group_ids_from_spec",
    "sample_noise_levels",
    "add_noise",
    "conditioned_input",
    "masked_score_matching_residual",
    "masked_denoising_score_matching_loss",
    "score_matching_loss",
    "lr_lambda_cosine_with_warmup",
    "EarlyStopping",
    "TrainHistory",
    "SimformerTrainer",
    "train_simformer",
    # constants
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_LR",
    "DEFAULT_GRAD_CLIP",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_WARMUP_STEPS",
    "DEFAULT_PATIENCE",
]

# ---------------------------------------------------------------------------
# defaults (Appendix A2.1)
# ---------------------------------------------------------------------------

DEFAULT_BATCH_SIZE = 1000
DEFAULT_LR = 3e-4  # paper only says "Adam"; 3e-4 with warmup + cosine is the documented default
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_MAX_STEPS = 100_000
DEFAULT_WARMUP_STEPS = 1000
DEFAULT_PATIENCE = 20
DEFAULT_MIN_LR_RATIO = 0.0

ArrayLike = Union[np.ndarray, "torch.Tensor"]


def _require_torch() -> None:
    if not _HAS_TORCH:  # pragma: no cover
        raise ImportError(
            "simformer.training requires PyTorch (pip install torch). Import error: "
            f"{_TORCH_IMPORT_ERROR}"
        )


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Training hyper-parameters (Appendix A2.1)."""

    batch_size: int = DEFAULT_BATCH_SIZE
    lr: float = DEFAULT_LR
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    gradient_clip: Optional[float] = DEFAULT_GRAD_CLIP

    max_steps: int = DEFAULT_MAX_STEPS
    max_epochs: Optional[int] = None
    warmup_steps: int = DEFAULT_WARMUP_STEPS
    lr_schedule: str = "cosine"  # "cosine" | "constant"
    min_lr_ratio: float = DEFAULT_MIN_LR_RATIO

    # objective
    loss_space: str = "score"  # "score" (Eq. 1-2) or "epsilon"
    lambda_t: Optional[float] = None  # optional scalar weighting of the per-sample loss

    # condition masks (Sec. 3.1 / A2.1)
    condition_mask_modes: Sequence[str] = tuple(DEFAULT_MODES)
    condition_mask_probs: Optional[Sequence[float]] = None
    p_random_low: float = P_RANDOM_LOW
    p_random_high: float = P_RANDOM_HIGH

    # validation / early stopping
    val_fraction: float = 0.1
    val_every: int = 500
    val_batches: int = 8
    early_stopping_patience: int = DEFAULT_PATIENCE
    early_stopping_min_delta: float = 1e-4

    log_every: int = 100
    seed: int = 0
    device: str = "cpu"
    dtype: str = "float32"
    normalize: bool = False
    grad_accumulation: int = 1

    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["condition_mask_modes"] = list(self.condition_mask_modes)
        d["betas"] = list(self.betas)
        if self.condition_mask_probs is not None:
            d["condition_mask_probs"] = list(self.condition_mask_probs)
        return d

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "TrainingConfig":
        cfg = dict(cfg or {})
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        extra = {k: cfg.pop(k) for k in list(cfg) if k not in known}
        if "betas" in cfg and cfg["betas"] is not None:
            cfg["betas"] = tuple(cfg["betas"])
        if "condition_mask_modes" in cfg and cfg["condition_mask_modes"] is not None:
            cfg["condition_mask_modes"] = tuple(cfg["condition_mask_modes"])
        if "condition_mask_probs" in cfg and cfg["condition_mask_probs"] is not None:
            cfg["condition_mask_probs"] = tuple(cfg["condition_mask_probs"])
        obj = cls(**cfg)
        if extra:
            obj.extra.update(extra)
        return obj

    @property
    def torch_dtype(self):
        _require_torch()
        return {"float32": torch.float32, "float64": torch.float64}[self.dtype]


# ---------------------------------------------------------------------------
# optional affine standardisation of the simulator output (sensible default,
# the paper does not specify one; disabled unless requested)
# ---------------------------------------------------------------------------


class Standardizer:
    """Per-dimension standardisation fitted on training pairs."""

    def __init__(self, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None):
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float64)
        self.std = None if std is None else np.asarray(std, dtype=np.float64)

    def fit(self, x: ArrayLike, eps: float = 1e-6) -> "Standardizer":
        arr = x.detach().cpu().numpy() if (_HAS_TORCH and hasattr(x, "detach")) else np.asarray(x)
        self.mean = arr.mean(axis=0)
        self.std = np.maximum(arr.std(axis=0), eps)
        return self

    def transform(self, x: ArrayLike) -> ArrayLike:
        if self.mean is None:
            return x
        if _HAS_TORCH and hasattr(x, "detach"):
            mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
            std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
            return (x - mean) / std
        return (np.asarray(x) - self.mean) / self.std

    def inverse_transform(self, x: ArrayLike) -> ArrayLike:
        if self.mean is None:
            return x
        if _HAS_TORCH and hasattr(x, "detach"):
            mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
            std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
            return x * std + mean
        return np.asarray(x) * self.std + self.mean

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": None if self.mean is None else self.mean.tolist(),
            "std": None if self.std is None else self.std.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Standardizer":
        return cls(mean=d.get("mean"), std=d.get("std"))


# ---------------------------------------------------------------------------
# mask helpers: variable level <-> value level
# ---------------------------------------------------------------------------


def _spec_of(tokenizer_or_spec: Any) -> Any:
    return getattr(tokenizer_or_spec, "spec", tokenizer_or_spec)


def token_group_ids_from_spec(tokenizer_or_spec: Any) -> np.ndarray:
    """Group id (statistical variable index) for every *value* of the joint vector.

    Ordering follows the tokenizer: ``[theta scalars | x scalars | function-valued values]``.
    """
    spec = _spec_of(tokenizer_or_spec)
    n_theta = len(getattr(spec, "parameter_names", ()) or ())
    n_data = len(getattr(spec, "data_names", ()) or ())
    groups: List[int] = []
    for i in range(n_theta + n_data):
        groups.append(i)
    for k, fspec in enumerate(getattr(spec, "function_valued", ()) or ()):
        n_points = getattr(fspec, "n_index_points", None)
        if n_points is None:
            n_points = int(np.asarray(fspec.index_set).reshape(-1).shape[0])
        groups.extend([n_theta + n_data + k] * int(n_points))
    return np.asarray(groups, dtype=int)


def variable_expansion_widths(tokenizer_or_spec: Any) -> np.ndarray:
    """Number of joint-vector entries each statistical variable contributes."""
    spec = _spec_of(tokenizer_or_spec)
    n_theta = len(getattr(spec, "parameter_names", ()) or ())
    n_data = len(getattr(spec, "data_names", ()) or ())
    widths = [1] * (n_theta + n_data)
    for fspec in getattr(spec, "function_valued", ()) or ():
        n_points = getattr(fspec, "n_index_points", None)
        if n_points is None:
            n_points = int(np.asarray(fspec.index_set).reshape(-1).shape[0])
        widths.append(int(n_points))
    return np.asarray(widths, dtype=int)


def expand_variable_mask(variable_mask: ArrayLike, tokenizer_or_spec: Any) -> ArrayLike:
    """Expand a ``(..., n_variables)`` mask to a ``(..., input_dim)`` mask."""
    widths = variable_expansion_widths(tokenizer_or_spec)
    if _HAS_TORCH and hasattr(variable_mask, "detach"):
        idx = torch.as_tensor(
            np.repeat(np.arange(len(widths)), widths), device=variable_mask.device, dtype=torch.long
        )
        return variable_mask[..., idx]
    mask = np.asarray(variable_mask)
    idx = np.repeat(np.arange(len(widths)), widths)
    return mask[..., idx]


def collapse_value_mask(value_mask: ArrayLike, tokenizer_or_spec: Any) -> ArrayLike:
    """Inverse of :func:`expand_variable_mask` using a max-reduction."""
    groups = token_group_ids_from_spec(tokenizer_or_spec)
    if _HAS_TORCH and hasattr(value_mask, "detach"):
        n_var = int(groups.max()) + 1 if groups.size else 0
        out = torch.zeros(value_mask.shape[:-1] + (n_var,), dtype=value_mask.dtype, device=value_mask.device)
        for g in range(n_var):
            sel = groups == g
            out[..., g] = value_mask[..., torch.as_tensor(sel, device=value_mask.device)].amax(dim=-1)
        return out
    mask = np.asarray(value_mask)
    n_var = int(groups.max()) + 1 if groups.size else 0
    out = np.zeros(mask.shape[:-1] + (n_var,), dtype=mask.dtype)
    for g in range(n_var):
        out[..., g] = mask[..., groups == g].max(axis=-1)
    return out


# ---------------------------------------------------------------------------
# noise / target helpers
# ---------------------------------------------------------------------------


def sample_noise_levels(
    batch_size: int, sde: SDE, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Uniformly sample the diffusion time in ``[t_min, t_max]`` (Sec. 3.3 / A2.1)."""
    rng = np.random.default_rng() if rng is None else rng
    t_min = float(getattr(sde, "t_min", 1e-5))
    t_max = float(getattr(sde, "t_max", 1.0))
    return rng.uniform(t_min, t_max, size=(int(batch_size),)).astype(np.float32)


def _marginal_mean_std(sde: SDE, t_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.asarray(sde.marginal_mean(t_np), dtype=np.float64).reshape(-1)
    sigma = np.asarray(sde.marginal_std(t_np), dtype=np.float64).reshape(-1)
    return mu, sigma


def add_noise(
    x0: ArrayLike,
    t: ArrayLike,
    sde: SDE,
    rng: Optional[np.random.Generator] = None,
    noise: Optional[ArrayLike] = None,
    standardized: bool = False,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """Forward-noising step ``x_t = mu(t) * x_0 + sigma(t) * eps``.

    Works for torch tensors (differentiable path) and numpy arrays.  ``t`` may be a
    ``(B,)`` array/tensor or a python float.  Returns ``(x_t, eps, mu, sigma)`` where the
    latter two have shape ``(B, 1)`` (broadcastable against ``x0``).
    """
    if _HAS_TORCH and hasattr(x0, "detach"):
        t_np = np.asarray(
            t.detach().cpu().numpy() if hasattr(t, "detach") else t, dtype=np.float64
        ).reshape(-1)
        if t_np.size == 1:
            t_np = np.repeat(t_np, x0.shape[0])
        mu_np, sigma_np = _marginal_mean_std(sde, t_np)
        mu = torch.as_tensor(mu_np, dtype=x0.dtype, device=x0.device).view(-1, 1)
        sigma = torch.as_tensor(sigma_np, dtype=x0.dtype, device=x0.device).view(-1, 1)
        if noise is None:
            noise = torch.randn_like(x0)
        x_t = mu * x0 + sigma * noise
        return x_t, noise, mu, sigma

    x0 = np.asarray(x0, dtype=np.float64)
    rng = np.random.default_rng() if rng is None else rng
    t_np = np.asarray(t, dtype=np.float64).reshape(-1)
    if t_np.size == 1:
        t_np = np.repeat(t_np, x0.shape[0])
    mu_np, sigma_np = _marginal_mean_std(sde, t_np)
    mu = mu_np.reshape(-1, 1)
    sigma = sigma_np.reshape(-1, 1)
    if noise is None:
        noise = rng.standard_normal(size=x0.shape)
    noise = np.asarray(noise, dtype=np.float64)
    x_t = mu * x0 + sigma * noise
    return x_t, noise, mu, sigma


def conditioned_input(x_t: ArrayLike, x0: ArrayLike, mask_values: ArrayLike) -> ArrayLike:
    """``(1 - M_C) * x_t + M_C * x_0`` (variables we condition on remain clean)."""
    return (1.0 - mask_values) * x_t + mask_values * x0


def masked_score_matching_residual(
    prediction: ArrayLike, target: ArrayLike, mask_tokens: ArrayLike
) -> ArrayLike:
    """``(1 - M_C) * (prediction - target)`` at token resolution."""
    return (1.0 - mask_tokens) * (prediction - target)


# ---------------------------------------------------------------------------
# the objective (Eq. 1-2)
# ---------------------------------------------------------------------------


def masked_denoising_score_matching_loss(
    model,
    sde: SDE,
    x0: ArrayLike,
    condition_mask: Optional[ArrayLike] = None,
    t: Optional[ArrayLike] = None,
    attention_mask: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    loss_space: str = "score",
    lambda_t: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    reduce: Optional[str] = "mean",
    return_parts: bool = False,
    function_values: Optional[ArrayLike] = None,
) -> Union[Any, Dict[str, Any]]:
    """Masked denoising score-matching loss of Sec. 3.3.

    Parameters
    ----------
    model:
        Either a ``SimformerScoreNetwork`` (tokenizer + transformer, called with the joint
        vector) or a bare ``TransformerScoreNetwork`` (in which case ``tokenizer`` must be
        provided).
    sde:
        SDE instance providing the marginal mean/std, i.e. the forward noising process.
    x0:
        Clean joint samples ``(B, input_dim)`` as torch tensor (differentiable) or numpy array.
    condition_mask:
        ``(B, n_variables)`` mask ``M_C`` (``1`` = conditioned on / kept clean).  If omitted an
        all-zero (joint) mask is used.
    t:
        Optional pre-sampled diffusion times ``(B,)``; uniformly sampled when omitted.
    attention_mask:
        ``M_E``: ``None``, a ``(n, n)`` mask, a ``(B, n, n)`` mask, or a callable
        ``M_C -> mask`` (used for condition-dependent masks obtained by graph inversion).
    loss_space:
        ``"score"`` reproduces Eq. 1 verbatim; ``"epsilon"`` is the equivalent eps-prediction
        parameterisation.

    Returns
    -------
    The scalar loss (or a dict with ``loss`` and diagnostics when ``return_parts=True``).
    """
    _require_torch()
    if loss_space not in ("score", "epsilon"):
        raise ValueError(f"loss_space must be 'score' or 'epsilon', got {loss_space!r}")

    if not hasattr(x0, "detach"):
        x0 = torch.as_tensor(np.asarray(x0), dtype=torch.float32)
    x0 = x0.float()
    batch_size = int(x0.shape[0])
    device, dtype = x0.device, x0.dtype

    # --- condition mask (variable level, then expanded to value level) ---------
    if condition_mask is None:
        mask_var = torch.zeros((batch_size, _n_variables(model, tokenizer)), device=device, dtype=dtype)
    else:
        mask_var = condition_mask if hasattr(condition_mask, "detach") else torch.as_tensor(
            np.asarray(condition_mask), dtype=dtype
        )
        mask_var = mask_var.to(device=device, dtype=dtype)
        if mask_var.dim() == 1:
            mask_var = mask_var.unsqueeze(0).expand(batch_size, -1)

    mask_values = _to_value_mask(mask_var, model, tokenizer, x0.shape[-1], device, dtype)

    # --- noise level ----------------------------------------------------------
    if t is None:
        t_np = sample_noise_levels(batch_size, sde, rng)
        t_t = torch.as_tensor(t_np, dtype=dtype, device=device)
    elif hasattr(t, "detach"):
        t_t = t.to(device=device, dtype=dtype).reshape(-1)
    else:
        t_t = torch.as_tensor(np.asarray(t, dtype=np.float32), dtype=dtype, device=device).reshape(-1)
        if t_t.numel() == 1:
            t_t = t_t.expand(batch_size)

    # --- forward noising and conditioning ------------------------------------
    x_t, noise, mu, sigma = add_noise(x0, t_t, sde)
    x_partial = conditioned_input(x_t, x0, mask_values)

    # --- score network --------------------------------------------------------
    attn = _resolve_attention_mask(attention_mask, mask_var, model, tokenizer, x_partial.shape[-2], device)
    prediction = _call_model(model, x_partial, t_t, mask_var, attn, tokenizer, function_values)
    # (B, n_tokens)
    if prediction.dim() == 3:
        if prediction.shape[-1] == 1:
            prediction = prediction[..., 0]
        else:
            prediction = prediction.reshape(prediction.shape[0], -1)
    if prediction.shape[1] != mask_var.shape[1] and tokenizer is not None:
        # model predicted at value resolution -> aggregate to token resolution is not possible;
        # instead expand the mask the same way.
        if prediction.shape[1] == mask_values.shape[1]:
            mask_tokens = mask_values
        else:
            raise ValueError(
                f"cannot align model output {tuple(prediction.shape)} with mask "
                f"{tuple(mask_var.shape)}"
            )
    else:
        mask_tokens = mask_var

    sigma_pred = sigma.view(-1, 1) if sigma.dim() == 2 else sigma.reshape(-1, 1)
    eps_target = noise
    score_target = -eps_target / sigma_pred

    if loss_space == "score":
        s_pred = -prediction / sigma_pred
        target = score_target
    else:
        s_pred = prediction
        target = eps_target

    residual = masked_score_matching_residual(s_pred, target, mask_tokens)
    per_sample = (residual ** 2).sum(dim=-1)
    if lambda_t is not None:
        per_sample = per_sample * float(lambda_t)

    if reduce is None:
        loss = per_sample
    elif reduce == "mean":
        loss = per_sample.mean()
    elif reduce == "sum":
        loss = per_sample.sum()
    else:
        raise ValueError(f"unknown reduce={reduce!r}")

    if not return_parts:
        return loss

    parts: Dict[str, Any] = {
        "loss": loss,
        "per_sample": per_sample.detach(),
        "residual": residual.detach(),
        "mask": mask_var.detach(),
        "t": t_t.detach(),
        "n_conditioned": mask_var.sum(dim=-1).detach(),
    }
    block_ids = _block_ids(model, tokenizer, mask_var.shape[1])
    if block_ids is not None:
        param_sel, data_sel = block_ids
        if param_sel.any() and data_sel.any():
            with torch.no_grad():
                parts["loss_params"] = (residual[:, param_sel] ** 2).sum(dim=-1).mean()
                parts["loss_data"] = (residual[:, data_sel] ** 2).sum(dim=-1).mean()
    return parts


def score_matching_loss(*args, **kwargs):
    """Alias of :func:`masked_denoising_score_matching_loss`."""
    return masked_denoising_score_matching_loss(*args, **kwargs)


# ---------------------------------------------------------------------------
# private plumbing
# ---------------------------------------------------------------------------


def _n_variables(model, tokenizer) -> int:
    tok = _get_tokenizer(model, tokenizer)
    if tok is None:
        raise ValueError("cannot infer the number of variables without a tokenizer")
    return int(getattr(tok, "n_variables", 0))


def _get_tokenizer(model, tokenizer):
    if tokenizer is not None:
        return tokenizer
    return getattr(model, "tokenizer", None)


def _to_value_mask(mask_var, model, tokenizer, n_values: int, device, dtype):
    tok = _get_tokenizer(model, tokenizer)
    if tok is not None and getattr(tok, "input_dim", None) == mask_var.shape[1]:
        return mask_var  # already at value resolution
    if tok is not None and getattr(tok, "n_variables", mask_var.shape[1]) == mask_var.shape[1]:
        widths = variable_expansion_widths(tok)
        if int(widths.sum()) == n_values:
            idx = np.repeat(np.arange(len(widths)), widths)
            return mask_var[:, torch.as_tensor(idx, device=mask_var.device)]
    if mask_var.shape[1] == n_values:
        return mask_var
    raise ValueError(
        f"condition mask of width {mask_var.shape[1]} cannot be expanded to {n_values} values"
    )


def _block_ids(model, tokenizer, n_tokens: int):
    tok = _get_tokenizer(model, tokenizer)
    spec = getattr(tok, "spec", None)
    if spec is None:
        return None
    n_theta = len(getattr(spec, "parameter_names", ()) or ())
    n_data = len(getattr(spec, "data_names", ()) or ())
    if n_theta + n_data == 0:
        return None
    group_ids = token_group_ids_from_spec(spec)
    if group_ids.size != n_tokens:
        return None
    param_sel = group_ids < n_theta
    data_sel = (group_ids >= n_theta) & (group_ids < n_theta + n_data)
    return (
        torch.as_tensor(param_sel, dtype=torch.bool),
        torch.as_tensor(data_sel, dtype=torch.bool),
    )


def _resolve_attention_mask(attention_mask, mask_var, model, tokenizer, n_tokens, device):
    if attention_mask is None:
        return None
    if callable(attention_mask):
        attn = attention_mask(mask_var.detach().cpu().numpy())
    else:
        attn = attention_mask
    if attn is None:
        return None
    if hasattr(attn, "detach"):  # already a tensor
        attn = attn.detach()
        return attn.to(device=device, dtype=torch.bool) if attn.dtype != torch.bool else attn.to(device)
    arr = np.asarray(attn)
    if arr.ndim == 2:
        return torch.as_tensor(arr > 0, dtype=torch.bool, device=device)
    if arr.ndim == 3:
        return torch.as_tensor(arr > 0, dtype=torch.bool, device=device)
    raise ValueError(f"unsupported attention mask shape {arr.shape}")


def _call_model(model, x_partial, t_t, mask_var, attn, tokenizer, function_values=None):
    # SimformerScoreNetwork(tokenizer, transformer): takes the joint vector directly.
    if hasattr(model, "tokenizer"):
        out = model(x_partial, t_t, condition_mask=mask_var, attention_mask=attn,
                    function_values=function_values)
    elif tokenizer is not None:
        tokens = tokenizer(x_partial, condition_mask=mask_var, function_values=function_values)
        out = model(tokens, t_t, attention_mask=attn)
    else:
        out = model(x_partial, t_t, attention_mask=attn)
    if isinstance(out, dict):
        out = out.get("score", out.get("eps", out.get("out")))
    return out


# ---------------------------------------------------------------------------
# optimisation utilities
# ---------------------------------------------------------------------------


def lr_lambda_cosine_with_warmup(
    step: int, warmup_steps: int, total_steps: int, min_ratio: float = 0.0, schedule: str = "cosine"
) -> float:
    """Linear warmup followed by cosine (or constant) decay."""
    step = int(step)
    warmup_steps = max(int(warmup_steps), 0)
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    if schedule != "cosine" or total_steps <= warmup_steps:
        return 1.0
    progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
    progress = min(max(progress, 0.0), 1.0)
    return float(min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


class EarlyStopping:
    """Early stopping on a monitored (validation) loss."""

    def __init__(self, patience: int = DEFAULT_PATIENCE, min_delta: float = 1e-4):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = math.inf
        self.best_step: Optional[int] = None
        self.counter = 0
        self.should_stop = False
        self.best_state: Optional[Dict[str, Any]] = None

    def update(self, value: float, step: Optional[int] = None, state: Optional[Dict[str, Any]] = None) -> bool:
        value = float(value)
        if value < self.best - self.min_delta:
            self.best = value
            self.best_step = step
            self.counter = 0
            if state is not None:
                self.best_state = _clone_state(state)
            return True
        self.counter += 1
        if self.patience >= 0 and self.counter >= self.patience:
            self.should_stop = True
        return False

    def reset(self) -> None:
        self.best = math.inf
        self.best_step = None
        self.counter = 0
        self.should_stop = False
        self.best_state = None


def _clone_state(state: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in state.items():
        if hasattr(v, "detach"):
            out[k] = {kk: vv.detach().clone() for kk, vv in v.items()} if isinstance(v, dict) else v.detach().clone()
        else:
            out[k] = v
    return out


@dataclass
class TrainHistory:
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    lr: List[float] = field(default_factory=list)
    step: List[int] = field(default_factory=list)
    val_step: List[int] = field(default_factory=list)
    elapsed: List[float] = field(default_factory=list)
    best_val_loss: Optional[float] = None
    best_step: Optional[int] = None
    stopped_early: bool = False
    epochs: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def log(self, kind: str, value) -> None:
        if kind in ("train", "train_loss"):
            self.train_loss.append(float(value))
        elif kind in ("val", "val_loss"):
            self.val_loss.append(float(value))


# ---------------------------------------------------------------------------
# trainer
# ---------------------------------------------------------------------------


class SimformerTrainer:
    """Trains a Simformer score network with masked denoising score matching.

    Example
    -------
    >>> trainer = SimformerTrainer(model, sde=get_sde("vesde"), attention_mask=M_E, tokenizer=tok)
    >>> history = trainer.fit(theta, x)
    """

    def __init__(
        self,
        model,
        sde: Optional[Union[SDE, str, Dict[str, Any], SDEConfig]] = None,
        attention_mask: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        config: Optional[Union[TrainingConfig, Dict[str, Any]]] = None,
        normalizer: Optional[Standardizer] = None,
        condition_masks: Optional[ConditionMaskSampler] = None,
        loss_space: Optional[str] = None,
    ):
        _require_torch()
        self.model = model
        self.sde = _resolve_sde(sde)
        self.attention_mask = attention_mask
        self.config = (
            config
            if isinstance(config, TrainingConfig)
            else TrainingConfig.from_dict(config or {})
        )
        self.tokenizer = _get_tokenizer(model, tokenizer)
        if self.tokenizer is None and not hasattr(model, "parameters"):
            raise ValueError("SimformerTrainer needs a torch module as `model`")

        self.device = _infer_device(model, self.config)
        self.model.to(self.device)
        if loss_space is not None:
            self.config.loss_space = loss_space

        # condition-mask sampling (Sec. 3.1 / A2.1)
        if condition_masks is not None:
            self.condition_masks = condition_masks
        else:
            n_var = int(getattr(self.tokenizer, "n_variables", 0)) if self.tokenizer is not None else 0
            n_par, n_data = _parameter_data_counts(self.tokenizer)
            self.condition_masks = ConditionMaskSampler(
                n_par,
                max(n_data, n_var - n_par) if n_var else n_data,
                modes=tuple(self.config.condition_mask_modes),
                p_random_low=self.config.p_random_low,
                p_random_high=self.config.p_random_high,
                probs=self.config.condition_mask_probs,
                seed=self.config.seed,
                device=self.device,
                dtype=self.config.torch_dtype,
            )

        self.rng = np.random.default_rng(self.config.seed)
        self.normalizer = normalizer
        if self.normalizer is None and self.config.normalize:
            self.normalizer = Standardizer()

        self.optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.lr,
            betas=tuple(self.config.betas),
            eps=self.config.eps,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = None
        self.global_step = 0
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            min_delta=self.config.early_stopping_min_delta,
        )
        self.history = TrainHistory()

    # -- data handling -----------------------------------------------------
    def joint(self, theta=None, x=None, x_joint=None) -> "torch.Tensor":
        if x_joint is not None:
            arr = x_joint
        else:
            if theta is None or x is None:
                raise ValueError("provide either `x_joint` or both `theta` and `x`")
            theta_t = theta if hasattr(theta, "detach") else torch.as_tensor(np.asarray(theta), dtype=torch.float32)
            x_t = x if hasattr(x, "detach") else torch.as_tensor(np.asarray(x), dtype=torch.float32)
            if theta_t.dim() == 1:
                theta_t = theta_t.unsqueeze(-1)
            if x_t.dim() == 1:
                x_t = x_t.unsqueeze(-1)
            arr = torch.cat([theta_t, x_t], dim=-1)
        arr = arr if hasattr(arr, "detach") else torch.as_tensor(np.asarray(arr), dtype=torch.float32)
        return arr.float().to(self.device)

    # -- loss --------------------------------------------------------------
    def loss(self, x0, condition_mask=None, t=None, return_parts: bool = False):
        return masked_denoising_score_matching_loss(
            self.model,
            self.sde,
            x0,
            condition_mask=condition_mask,
            t=t,
            attention_mask=self.attention_mask,
            tokenizer=self.tokenizer,
            loss_space=self.config.loss_space,
            lambda_t=self.config.lambda_t,
            rng=self.rng,
            return_parts=return_parts,
        )

    def sample_condition_masks(self, batch_size: int):
        return self.condition_masks.sample(int(batch_size))

    # -- step --------------------------------------------------------------
    def train_step(self, x0, condition_mask=None) -> Dict[str, float]:
        self.model.train()
        if condition_mask is None:
            condition_mask = self.sample_condition_masks(int(x0.shape[0]))
        parts = self.loss(x0, condition_mask=condition_mask, return_parts=True)
        loss = parts["loss"] / max(self.config.grad_accumulation, 1)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                float(self.config.gradient_clip),
            )
        lr = self._current_lr()
        self.optimizer.step()
        self.global_step += 1
        metrics = {
            "loss": float(parts["loss"].detach().cpu()),
            "lr": float(lr),
            "n_conditioned": float(parts["n_conditioned"].float().mean().cpu()),
        }
        for key in ("loss_params", "loss_data"):
            if key in parts:
                metrics[key] = float(parts[key].detach().cpu())
        return metrics

    def _current_lr(self) -> float:
        if self.scheduler is not None:
            return float(self.scheduler.get_last_lr()[0])
        return float(
            self.config.lr
            * lr_lambda_cosine_with_warmup(
                self.global_step,
                self.config.warmup_steps,
                max(int(self.config.max_steps), 1),
                self.config.min_lr_ratio,
                self.config.lr_schedule,
            )
        )

    def _build_scheduler(self) -> None:
        total = max(int(self.config.max_steps), 1)
        lam = lambda step: lr_lambda_cosine_with_warmup(  # noqa: E731
            step, self.config.warmup_steps, total, self.config.min_lr_ratio, self.config.lr_schedule
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lam)

    # -- validation --------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, x_val, condition_mask=None, n_batches: Optional[int] = None) -> float:
        self.model.eval()
        if condition_mask is None and self.config.val_every > 0:
            # fixed validation masks keep the metric comparable across steps
            condition_mask = self.condition_masks.all_modes()
            if condition_mask is not None and not hasattr(condition_mask.get(list(condition_mask)[0]), "shape"):
                condition_mask = None
        losses = []
        n_batches = int(n_batches or self.config.val_batches)
        bs = int(self.config.batch_size)
        n = int(x_val.shape[0])
        if n == 0:
            return float("nan")
        for b in range(max(n_batches, 1)):
            idx = self.rng.integers(0, n, size=min(bs, n))
            xb = x_val[torch.as_tensor(idx, device=x_val.device)]
            losses.append(float(self.loss(xb).detach().cpu()))
        return float(np.mean(losses)) if losses else float("nan")

    # -- main loop ---------------------------------------------------------
    def fit(
        self,
        theta=None,
        x=None,
        x_joint=None,
        val_theta=None,
        val_x=None,
        val_joint=None,
        val_fraction: Optional[float] = None,
        callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Train the model; returns the history dictionary."""
        data = self.joint(theta, x, x_joint)
        if data.dim() == 1:
            data = data.unsqueeze(-1)

        # optional standardisation
        if self.normalizer is not None and self.normalizer.mean is None:
            self.normalizer.fit(data.detach().cpu().numpy())
        if self.normalizer is not None:
            data = self.normalizer.transform(data)

        # train/val split
        if val_joint is not None or val_theta is not None:
            val = self.joint(val_theta, val_x, val_joint)
            if self.normalizer is not None:
                val = self.normalizer.transform(val)
            train = data
        else:
            vf = self.config.val_fraction if val_fraction is None else float(val_fraction)
            n = int(data.shape[0])
            n_val = int(round(n * vf)) if vf and vf > 0 else 0
            if n_val > 0:
                perm = torch.as_tensor(self.rng.permutation(n), device=data.device)
                val = data[perm[:n_val]]
                train = data[perm[n_val:]]
            else:
                train, val = data, None

        n_train = int(train.shape[0])
        bs = min(int(self.config.batch_size), max(n_train, 1))
        steps_per_epoch = max(1, n_train // bs)
        if self.config.max_epochs is not None:
            max_steps = min(int(self.config.max_steps), steps_per_epoch * int(self.config.max_epochs))
        else:
            max_steps = int(self.config.max_steps)
        self._build_scheduler()

        # fixed validation masks (joint / posterior / likelihood / random)
        val_masks = self.condition_masks.all_modes()
        val_mask_list = list(val_masks.values())
        val_mask = val_mask_list[0] if val_mask_list else None
        if val_mask is not None:
            val_mask = val_mask.expand(self.config.batch_size, -1)

        start = time.time()
        epoch = 0
        while self.global_step < max_steps and not self.early_stopping.should_stop:
            epoch += 1
            perm = torch.as_tensor(self.rng.permutation(n_train), device=train.device)
            for b in range(steps_per_epoch):
                if self.global_step >= max_steps or self.early_stopping.should_stop:
                    break
                idx = perm[b * bs : (b + 1) * bs]
                if idx.numel() == 0:
                    continue
                metrics = self.train_step(train[idx])
                self.history.train_loss.append(metrics["loss"])
                self.history.lr.append(metrics["lr"])
                self.history.step.append(self.global_step)
                self.history.elapsed.append(time.time() - start)
                if self.scheduler is not None:
                    self.scheduler.step()

                if self.config.log_every and self.global_step % self.config.log_every == 0 and verbose:
                    print(
                        f"[simformer] step {self.global_step:6d} loss {metrics['loss']:.4f} "
                        f"lr {metrics['lr']:.2e}"
                    )
                if callback is not None:
                    callback(self.global_step, metrics)

                if (
                    val is not None
                    and self.config.val_every
                    and self.global_step % self.config.val_every == 0
                ):
                    vloss = self.evaluate(val, condition_mask=val_mask)
                    self.history.val_loss.append(vloss)
                    self.history.val_step.append(self.global_step)
                    improved = self.early_stopping.update(
                        vloss,
                        step=self.global_step,
                        state={"model": self.model.state_dict()},
                    )
                    if improved:
                        self.history.best_val_loss = vloss
                        self.history.best_step = self.global_step
                    if verbose:
                        print(f"[simformer] step {self.global_step:6d} val {vloss:.4f}")
        self.history.epochs = epoch
        self.history.stopped_early = bool(self.early_stopping.should_stop)

        if self.early_stopping.best_state is not None and "model" in self.early_stopping.best_state:
            self.model.load_state_dict(self.early_stopping.best_state["model"])
        return self.history.to_dict()

    # -- checkpoints -------------------------------------------------------
    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "global_step": self.global_step,
            "history": self.history.to_dict(),
            "sde": getattr(self.sde, "name", None),
            "sde_config": _sde_to_dict(self.sde),
            "normalizer": None if self.normalizer is None else self.normalizer.to_dict(),
        }
        torch.save(payload, path)
        return path

    def load(self, path: str, strict: bool = True, load_optimizer: bool = True) -> "SimformerTrainer":
        payload = torch.load(path, map_location=self.device)
        self.model.load_state_dict(payload["model"], strict=strict)
        if load_optimizer and "optimizer" in payload:
            try:
                self.optimizer.load_state_dict(payload["optimizer"])
            except Exception:
                pass
        self.global_step = int(payload.get("global_step", 0))
        if payload.get("normalizer"):
            self.normalizer = Standardizer.from_dict(payload["normalizer"])
        hist = payload.get("history")
        if hist:
            self.history = TrainHistory(**{k: v for k, v in hist.items()
                                           if k in TrainHistory.__dataclass_fields__})
        return self


# ---------------------------------------------------------------------------
# convenience entry points
# ---------------------------------------------------------------------------


def train_simformer(
    model,
    theta=None,
    x=None,
    x_joint=None,
    attention_mask: Optional[Any] = None,
    sde: Optional[Union[SDE, str, Dict[str, Any]]] = None,
    config: Optional[Union[TrainingConfig, Dict[str, Any]]] = None,
    verbose: bool = False,
    **fit_kwargs,
) -> Tuple[SimformerTrainer, Dict[str, Any]]:
    """Build a trainer and run :meth:`SimformerTrainer.fit`."""
    trainer = SimformerTrainer(model, sde=sde, attention_mask=attention_mask, config=config)
    history = trainer.fit(theta, x, x_joint, verbose=verbose, **fit_kwargs)
    return trainer, history


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _resolve_sde(sde) -> SDE:
    if sde is None:
        return get_sde("vesde")
    if isinstance(sde, SDE):
        return sde
    if isinstance(sde, str):
        return get_sde(sde)
    if isinstance(sde, SDEConfig):
        return sde.build()
    if isinstance(sde, dict):
        return sde_from_config(sde)
    # duck-typed SDE (has marginal_mean / marginal_std)
    if hasattr(sde, "marginal_mean") and hasattr(sde, "marginal_std"):
        return sde
    raise TypeError(f"cannot interpret SDE specification {sde!r}")


def _sde_to_dict(sde: SDE) -> Dict[str, Any]:
    out = {"name": getattr(sde, "name", None)}
    for key in ("sigma_max", "sigma_min", "beta_min", "beta_max", "t_min", "t_max", "n_steps"):
        if hasattr(sde, key):
            out[key] = getattr(sde, key)
    return out


def _infer_device(model, config: TrainingConfig):
    try:
        p = next(model.parameters())
        return p.device
    except Exception:
        return torch.device(config.device)


def _parameter_data_counts(tokenizer) -> Tuple[int, int]:
    spec = getattr(tokenizer, "spec", None)
    if spec is None:
        n_var = int(getattr(tokenizer, "n_variables", 0))
        n_par = int(getattr(tokenizer, "n_parameter_variables", n_var))
        return n_par, max(n_var - n_par, 0)
    n_par = len(getattr(spec, "parameter_names", ()) or ()) + len(
        getattr(spec, "function_valued", ()) or ()
    )
    n_data = len(getattr(spec, "data_names", ()) or ())
    return int(n_par), int(n_data)
