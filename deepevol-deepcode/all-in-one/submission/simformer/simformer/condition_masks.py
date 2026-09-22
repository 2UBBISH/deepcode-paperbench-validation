"""Per-sample condition masks ``M_C`` for the Simformer.

Paper references
----------------
Sec. 3.1 ("A Tokenizer for SBI"):
    "The condition state is a binary variable and signifies whether the variable is
    conditioned on or not. It is resampled for every (theta, x) in R^d pair at every
    iteration of training. We denote the condition state of all variables as
    M_C in {0,1}^d. Setting M_C = (0, ..., 0) corresponds to an unconditional diffusion
    model, whereas adopting M_C^{(i)} = 1 for data and M_C^{(i)} = 0 for parameters
    corresponds to training a conditional diffusion model of the posterior ...
    In our experiments, we uniformly at random sample either the masks for the joint,
    the posterior, the likelihood, or two randomly sampled masks."

Sec. 3.3 ("Simformer training and sampling"):
    x_hat_t^{M_C} = (1 - M_C) * x_hat_t + M_C * x_hat_0            (partially noisy input)
    ell = (1 - M_C) * (s_phi^{M_E}(x_hat_t^{M_C}, t) - grad log p_t(x_hat_t | x_hat_0))
    L   = E[ ||ell||_2^2 ]

Sec. A2.1 ("Training and model configurations"):
    "At every training batch, we selected uniformly at random a mask corresponding to
     the joint, the posterior, the likelihood or two random masks. The random masks were
     drawn from a Bernoulli distribution with p = 0.3 and p = 0.7."

Conventions
-----------
* ``M_C[i] = 1``  -> token ``i`` is *conditioned on* (observed / clamped), it stays clean
  at ``x_hat_0`` during training and is fixed at its conditioning value during sampling.
* ``M_C[i] = 0``  -> token ``i`` is *latent*; the score model is trained to predict its
  score and the reverse SDE is run on it.
* Token ordering is the tokenizer ordering:
  ``[theta scalars | x scalars | function-valued parameter tokens]``
  (see :mod:`simformer.tokenizer`).  Therefore the parameter block is the prefix
  ``[0, n_parameters)`` and the data block is the suffix ``[n_parameters, d)``.

Function-valued parameters occupy several tokens that belong to one statistical
variable.  Such tokens must be conditioned coherently (all-or-none).  This module
supports that through an optional ``group_ids`` vector mapping each token to the
index of the statistical variable it belongs to; random masks are then drawn
per *variable* and broadcast to the tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:  # pragma: no cover - torch is the neural-network backend used by default
    import torch
    from torch import Tensor

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    Tensor = None  # type: ignore[assignment]
    _HAS_TORCH = False

from .utils import as_mask

ArrayLike = Union[np.ndarray, "Tensor"]

# --------------------------------------------------------------------------------------
# Mask-mode constants
# --------------------------------------------------------------------------------------

#: ``M_C = (0, ..., 0)``: unconditional diffusion model over the joint distribution.
JOINT = "joint"
#: ``M_C = 1`` for data, ``0`` for parameters: conditional model of the posterior p(theta|x).
POSTERIOR = "posterior"
#: ``M_C = 1`` for parameters, ``0`` for data: conditional model of the likelihood p(x|theta).
LIKELIHOOD = "likelihood"
#: Random mask with ``M_C^{(i)} ~ Bernoulli(0.3)``.
RANDOM_LOW = "random_low"
#: Random mask with ``M_C^{(i)} ~ Bernoulli(0.7)``.
RANDOM_HIGH = "random_high"

#: The five mask types used during Simformer training (paper Sec. 3.1 / A2.1).
DEFAULT_MODES: Tuple[str, ...] = (JOINT, POSTERIOR, LIKELIHOOD, RANDOM_LOW, RANDOM_HIGH)

#: Bernoulli probabilities of the two random mask types.
P_RANDOM_LOW = 0.3
P_RANDOM_HIGH = 0.7

_MODE_ALIASES: Dict[str, str] = {
    "joint": JOINT,
    "unconditional": JOINT,
    "posterior": POSTERIOR,
    "params": POSTERIOR,
    "likelihood": LIKELIHOOD,
    "data": LIKELIHOOD,
    "random": RANDOM_LOW,
    "random_low": RANDOM_LOW,
    "random_0.3": RANDOM_LOW,
    "random_high": RANDOM_HIGH,
    "random_0.7": RANDOM_HIGH,
}


def normalize_mode(mode: str) -> str:
    """Map user supplied aliases onto the canonical mask-mode names."""
    key = str(mode).strip().lower()
    if key not in _MODE_ALIASES:
        raise ValueError(
            f"Unknown condition-mask mode {mode!r}; expected one of {sorted(set(_MODE_ALIASES.values()))}."
        )
    return _MODE_ALIASES[key]


# --------------------------------------------------------------------------------------
# Deterministic mask builders (numpy, float32 in {0, 1} as in the paper)
# --------------------------------------------------------------------------------------


def _new_rng(rng: Optional[np.random.Generator] = None) -> np.random.Generator:
    if rng is None:
        return np.random.default_rng()
    return rng


def joint_mask(n_variables: int, batch_size: Optional[int] = None) -> np.ndarray:
    """``M_C = (0, ..., 0)`` -- unconditional joint model (Sec. 3.1)."""
    if batch_size is None:
        return np.zeros(int(n_variables), dtype=np.float32)
    return np.zeros((int(batch_size), int(n_variables)), dtype=np.float32)


def posterior_condition_mask(n_parameters: int, n_data: int) -> np.ndarray:
    """``M_C^{(i)} = 1`` for data and ``0`` for parameters -> posterior p(theta | x)."""
    return np.concatenate(
        [
            np.zeros(int(n_parameters), dtype=np.float32),
            np.ones(int(n_data), dtype=np.float32),
        ]
    )


def likelihood_condition_mask(n_parameters: int, n_data: int) -> np.ndarray:
    """``M_C^{(i)} = 1`` for parameters and ``0`` for data -> likelihood p(x | theta)."""
    return np.concatenate(
        [
            np.ones(int(n_parameters), dtype=np.float32),
            np.zeros(int(n_data), dtype=np.float32),
        ]
    )


def random_condition_mask(
    n_parameters: int,
    n_data: int,
    p: float = P_RANDOM_LOW,
    rng: Optional[np.random.Generator] = None,
    batch_size: Optional[int] = None,
) -> np.ndarray:
    """Bernoulli ``p`` mask over all variables (paper: p = 0.3 and p = 0.7)."""
    rng = _new_rng(rng)
    d = int(n_parameters) + int(n_data)
    if batch_size is None:
        return (rng.random(d) < float(p)).astype(np.float32)
    return (rng.random((int(batch_size), d)) < float(p)).astype(np.float32)


def mode_mask(
    mode: str,
    n_parameters: int,
    n_data: int,
    rng: Optional[np.random.Generator] = None,
    *,
    p_low: float = P_RANDOM_LOW,
    p_high: float = P_RANDOM_HIGH,
) -> np.ndarray:
    """Build the (single-sample) mask corresponding to one canonical ``mode``."""
    mode = normalize_mode(mode)
    if mode == JOINT:
        return joint_mask(int(n_parameters) + int(n_data))
    if mode == POSTERIOR:
        return posterior_condition_mask(n_parameters, n_data)
    if mode == LIKELIHOOD:
        return likelihood_condition_mask(n_parameters, n_data)
    if mode == RANDOM_LOW:
        return random_condition_mask(n_parameters, n_data, p_low, rng)
    if mode == RANDOM_HIGH:
        return random_condition_mask(n_parameters, n_data, p_high, rng)
    raise ValueError(f"Unhandled mask mode {mode!r}.")  # pragma: no cover


def group_broadcast(
    variable_mask: np.ndarray, group_ids: Optional[np.ndarray], n_variables: int
) -> np.ndarray:
    """Broadcast a per-(statistical) variable mask back to the token level.

    Function-valued parameters occupy several tokens that share one statistical
    variable; this function expands a mask defined over variables to a mask defined
    over tokens.
    """
    if group_ids is None:
        return np.asarray(variable_mask, dtype=np.float32)
    group_ids = np.asarray(group_ids).astype(np.int64).reshape(-1)
    if group_ids.shape[0] != int(n_variables):
        raise ValueError(
            f"group_ids has length {group_ids.shape[0]} but there are {n_variables} tokens."
        )
    arr = np.asarray(variable_mask, dtype=np.float32)
    if arr.ndim == 1:
        return arr[group_ids]
    return arr[:, group_ids]


def group_collapse(mask: np.ndarray, group_ids: Optional[np.ndarray]) -> np.ndarray:
    """Inverse of :func:`group_broadcast`: one value per statistical variable.

    A statistical variable counts as conditioned if *all* of its tokens are conditioned.
    """
    if group_ids is None:
        return np.asarray(mask, dtype=np.float32)
    group_ids = np.asarray(group_ids).astype(np.int64).reshape(-1)
    arr = np.asarray(mask, dtype=np.float32)
    n_groups = int(group_ids.max()) + 1 if group_ids.size else 0
    if arr.ndim == 1:
        out = np.zeros(n_groups, dtype=np.float32)
        for g in range(n_groups):
            out[g] = float(arr[group_ids == g].min())
        return out
    out = np.zeros((arr.shape[0], n_groups), dtype=np.float32)
    for g in range(n_groups):
        out[:, g] = arr[:, group_ids == g].min(axis=1)
    return out


# --------------------------------------------------------------------------------------
# Partially-noised inputs and loss weighting (Sec. 3.3)
# --------------------------------------------------------------------------------------


def apply_condition(x_t: ArrayLike, x_0: ArrayLike, condition_mask: ArrayLike) -> ArrayLike:
    """``x_hat_t^{M_C} = (1 - M_C) * x_hat_t + M_C * x_hat_0`` (Sec. 3.3).

    Conditioned variables remain clean, latent variables stay noisy.  Works with both
    NumPy arrays and torch tensors (the backend of the two first arguments decides).
    """
    if _HAS_TORCH and isinstance(x_t, torch.Tensor):
        m = torch.as_tensor(
            np.asarray(condition_mask), dtype=x_t.dtype, device=x_t.device
        )
        x0 = x_0 if isinstance(x_0, torch.Tensor) else torch.as_tensor(x_0, dtype=x_t.dtype, device=x_t.device)
        return (1.0 - m) * x_t + m * x0
    m = np.asarray(condition_mask, dtype=np.asarray(x_t).dtype)
    return (1.0 - m) * np.asarray(x_t) + m * np.asarray(x_0)


def partially_noise(x_t: ArrayLike, x_0: ArrayLike, condition_mask: ArrayLike) -> ArrayLike:
    """Alias of :func:`apply_condition` (paper's partially noisy sample)."""
    return apply_condition(x_t, x_0, condition_mask)


def loss_mask(condition_mask: ArrayLike) -> ArrayLike:
    """``(1 - M_C)``: weight applied to the score-matching residual (Sec. 3.3)."""
    if _HAS_TORCH and isinstance(condition_mask, torch.Tensor):
        return 1.0 - condition_mask
    return 1.0 - np.asarray(condition_mask, dtype=np.float32)


def masked_residual(
    score: ArrayLike, target_score: ArrayLike, condition_mask: ArrayLike
) -> ArrayLike:
    """``ell = (1 - M_C) * (s_phi(x_hat_t^{M_C}, t) - grad log p_t(x_hat_t | x_hat_0))``."""
    if _HAS_TORCH and isinstance(score, torch.Tensor):
        m = torch.as_tensor(np.asarray(condition_mask), dtype=score.dtype, device=score.device)
        return (1.0 - m) * (score - target_score)
    m = np.asarray(condition_mask, dtype=np.float32)
    return (1.0 - m) * (np.asarray(score) - np.asarray(target_score))


def score_matching_loss(
    score: ArrayLike, target_score: ArrayLike, condition_mask: ArrayLike, *, reduce: bool = True
) -> ArrayLike:
    """``L = E[ || (1 - M_C) * (s_phi - grad log p_t) ||_2^2 ]`` (Sec. 3.3, Eq. 2)."""
    residual = masked_residual(score, target_score, condition_mask)
    if _HAS_TORCH and isinstance(residual, torch.Tensor):
        per_sample = residual.reshape(residual.shape[0], -1).pow(2).sum(dim=-1)
        return per_sample.mean() if reduce else per_sample
    residual = np.asarray(residual, dtype=np.float64)
    per_sample = (residual.reshape(residual.shape[0], -1) ** 2).sum(axis=-1)
    return float(per_sample.mean()) if reduce else per_sample


def conditioned_values(
    x_0: ArrayLike, condition_mask: ArrayLike, fill: float = 0.0
) -> ArrayLike:
    """Return the conditioning values ``M_C * x_hat_0`` (zeros for latent variables)."""
    if _HAS_TORCH and isinstance(x_0, torch.Tensor):
        m = torch.as_tensor(np.asarray(condition_mask), dtype=x_0.dtype, device=x_0.device)
        return m * x_0
    m = np.asarray(condition_mask, dtype=np.float32)
    return m * np.asarray(x_0) if fill == 0.0 else m * np.asarray(x_0) + (1.0 - m) * fill


# --------------------------------------------------------------------------------------
# Index helpers
# --------------------------------------------------------------------------------------


def condition_indices(condition_mask: ArrayLike) -> np.ndarray:
    """Indices of conditioned (observed) variables, taken from the first batch row."""
    m = np.asarray(condition_mask)
    if m.ndim > 1:
        m = m[0]
    return np.nonzero(m.astype(bool))[0]


def latent_indices(condition_mask: ArrayLike) -> np.ndarray:
    """Indices of latent (unobserved) variables, taken from the first batch row."""
    m = np.asarray(condition_mask)
    if m.ndim > 1:
        m = m[0]
    return np.nonzero(~m.astype(bool))[0]


def merge_condition_masks(*masks: ArrayLike) -> np.ndarray:
    """Union of condition masks: a variable is conditioned if any input says so."""
    if not masks:
        raise ValueError("merge_condition_masks requires at least one mask.")
    out = as_mask(masks[0])
    for m in masks[1:]:
        out = np.maximum(out, as_mask(m))
    return out.astype(np.float32)


def validate_condition_mask(mask: ArrayLike, n_variables: Optional[int] = None) -> np.ndarray:
    """Check that a mask is binary, finite and of the right width."""
    arr = np.asarray(mask, dtype=np.float32)
    if arr.ndim not in (1, 2):
        raise ValueError(f"condition mask must be 1-D or 2-D, got shape {arr.shape}.")
    if n_variables is not None:
        width = arr.shape[-1]
        if width != int(n_variables):
            raise ValueError(
                f"condition mask has width {width} but {n_variables} variables were expected."
            )
    if not np.all(np.isfinite(arr)):
        raise ValueError("condition mask contains non-finite entries.")
    if not np.all((arr == 0.0) | (arr == 1.0)):
        raise ValueError("condition mask must be binary (entries in {0, 1}).")
    return arr


# --------------------------------------------------------------------------------------
# Mask sampler
# --------------------------------------------------------------------------------------


@dataclass
class ConditionMaskConfig:
    """Configuration of the training-time distribution over condition masks.

    Parameters
    ----------
    n_parameters:
        Number of parameter *tokens* (including all function-valued tokens).
    n_data:
        Number of data *tokens*.
    modes:
        Mask types that are sampled uniformly at random each batch element
        (default: joint, posterior, likelihood, Bernoulli(0.3), Bernoulli(0.7)).
    p_random_low, p_random_high:
        Bernoulli probabilities of the two random mask types.
    group_ids:
        Optional length-``n_parameters + n_data`` vector mapping each token onto the
        statistical variable it belongs to.  Random masks are drawn per variable and
        broadcast to the tokens, so that function-valued parameters are conditioned
        coherently.
    probs:
        Optional explicit sampling probabilities for ``modes`` (must sum to 1).
    """

    n_parameters: int
    n_data: int
    modes: Sequence[str] = field(default_factory=lambda: tuple(DEFAULT_MODES))
    p_random_low: float = P_RANDOM_LOW
    p_random_high: float = P_RANDOM_HIGH
    group_ids: Optional[np.ndarray] = None
    probs: Optional[Sequence[float]] = None

    @property
    def n_variables(self) -> int:
        return int(self.n_parameters) + int(self.n_data)

    @property
    def n_groups(self) -> int:
        if self.group_ids is None:
            return self.n_variables
        return int(np.asarray(self.group_ids).max()) + 1

    def group_mask(self) -> np.ndarray:
        """Group ids at the statistical-variable level (identity when not grouped)."""
        if self.group_ids is None:
            return np.arange(self.n_variables, dtype=np.int64)
        return np.asarray(self.group_ids).astype(np.int64).reshape(-1)


class ConditionMaskSampler:
    """Draws per-sample condition masks ``M_C`` exactly as described in the paper.

    Example
    -------
    >>> sampler = ConditionMaskSampler(n_parameters=5, n_data=8)
    >>> M_C = sampler.sample(batch_size=1000)        # (1000, 13) tensor of 0/1
    >>> M_C, modes = sampler.sample_with_modes(1000) # also returns the chosen modes
    """

    def __init__(
        self,
        n_parameters: int,
        n_data: int,
        *,
        modes: Sequence[str] = DEFAULT_MODES,
        p_random_low: float = P_RANDOM_LOW,
        p_random_high: float = P_RANDOM_HIGH,
        probs: Optional[Sequence[float]] = None,
        group_ids: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        requires_grad: bool = False,
        device: Optional[Union[str, "torch.device"]] = None,
        dtype: Optional["torch.dtype"] = None,
    ) -> None:
        self.config = ConditionMaskConfig(
            n_parameters=int(n_parameters),
            n_data=int(n_data),
            modes=tuple(normalize_mode(m) for m in modes),
            p_random_low=float(p_random_low),
            p_random_high=float(p_random_high),
            group_ids=None if group_ids is None else np.asarray(group_ids).astype(np.int64),
            probs=None if probs is None else tuple(float(p) for p in probs),
        )
        if self.config.probs is not None:
            if len(self.config.probs) != len(self.config.modes):
                raise ValueError("probs must have the same length as modes.")
            total = float(sum(self.config.probs))
            if total <= 0:
                raise ValueError("probs must sum to a positive value.")
        self.rng = np.random.default_rng(seed)
        self.requires_grad = bool(requires_grad)
        self.device = device
        self.dtype = dtype

    # -- properties ------------------------------------------------------------------
    @property
    def n_variables(self) -> int:
        return self.config.n_variables

    @property
    def n_parameters(self) -> int:
        return self.config.n_parameters

    @property
    def n_data(self) -> int:
        return self.config.n_data

    @property
    def n_groups(self) -> int:
        return self.config.n_groups

    @property
    def modes(self) -> Tuple[str, ...]:
        return tuple(self.config.modes)

    # -- numpy / list interface -------------------------------------------------------
    def _probs(self) -> np.ndarray:
        if self.config.probs is None:
            return np.full(len(self.config.modes), 1.0 / len(self.config.modes))
        p = np.asarray(self.config.probs, dtype=np.float64)
        return p / p.sum()

    def choose_modes(self, batch_size: int) -> List[str]:
        """Sample one mode (joint/posterior/likelihood/random) per batch element."""
        idx = self.rng.choice(len(self.config.modes), size=int(batch_size), p=self._probs())
        modes = [self.config.modes[int(i)] for i in np.atleast_1d(idx)]
        return modes

    def sample_numpy(
        self, batch_size: int = 1, *, return_modes: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[str]]]:
        """Draw ``M_C`` as a ``(batch_size, d)`` float32 NumPy array."""
        batch_size = int(batch_size)
        modes = self.choose_modes(batch_size)
        d = self.n_variables
        group_ids = self.config.group_mask()
        n_groups = self.n_groups
        out = np.zeros((batch_size, d), dtype=np.float32)
        for b, mode in enumerate(modes):
            if mode == JOINT:
                var_mask = np.zeros(n_groups, dtype=np.float32)
            elif mode == POSTERIOR:
                var_mask = np.concatenate(
                    [
                        np.zeros(self.n_parameters, dtype=np.float32),
                        np.ones(self.n_data, dtype=np.float32),
                    ]
                )
            elif mode == LIKELIHOOD:
                var_mask = np.concatenate(
                    [
                        np.ones(self.n_parameters, dtype=np.float32),
                        np.zeros(self.n_data, dtype=np.float32),
                    ]
                )
            else:
                p = (
                    self.config.p_random_low
                    if mode == RANDOM_LOW
                    else self.config.p_random_high
                )
                var_mask = (self.rng.random(n_groups) < p).astype(np.float32)
            out[b] = group_broadcast(var_mask, self.config.group_ids, d)
        if return_modes:
            return out, modes
        return out

    def sample(
        self, batch_size: int = 1, *, return_modes: bool = False
    ) -> Union["Tensor", Tuple["Tensor", List[str]]]:
        """Draw ``M_C`` as a torch tensor (falls back to NumPy when torch is missing)."""
        arr, modes = self.sample_numpy(batch_size, return_modes=True)  # type: ignore[misc]
        if not _HAS_TORCH:
            return (arr, modes) if return_modes else arr  # type: ignore[return-value]
        tensor = torch.as_tensor(arr, device=self.device, dtype=self.dtype or torch.float32)
        if self.requires_grad:
            tensor.requires_grad_(True)
        return (tensor, modes) if return_modes else tensor

    # -- individual modes -------------------------------------------------------------
    def single_mode(self, mode: str, batch_size: int = 1) -> "Tensor":
        """Draw ``batch_size`` copies of one specific mask type (no random mixture)."""
        mode = normalize_mode(mode)
        arr = np.stack([mode_mask(mode, self.n_parameters, self.n_data, self.rng,
                                  p_low=self.config.p_random_low,
                                  p_high=self.config.p_random_high)
                        for _ in range(int(batch_size))])
        if self.config.group_ids is not None and mode in (RANDOM_LOW, RANDOM_HIGH):
            # re-draw grouped random masks coherently at the variable level
            p = self.config.p_random_low if mode == RANDOM_LOW else self.config.p_random_high
            var = (self.rng.random((int(batch_size), self.n_groups)) < p).astype(np.float32)
            arr = group_broadcast(var, self.config.group_ids, self.n_variables)
        if not _HAS_TORCH:
            return arr  # type: ignore[return-value]
        return torch.as_tensor(arr, device=self.device, dtype=self.dtype or torch.float32)

    # -- convenience -------------------------------------------------------------------
    def all_modes(self) -> Dict[str, "Tensor"]:
        """One mask of each configured mode (useful for evaluation / tests)."""
        return {m: self.single_mode(m, 1)[0] for m in self.modes}

    def conditional_kwargs(self, condition_mask: ArrayLike) -> Dict[str, np.ndarray]:
        """Latent / observed index sets for use by samplers and guidance."""
        mask = validate_condition_mask(condition_mask, self.n_variables)
        if mask.ndim > 1:
            mask = mask[0]
        return {
            "condition_mask": mask,
            "condition_indices": np.nonzero(mask.astype(bool))[0],
            "latent_indices": np.nonzero(~mask.astype(bool))[0],
        }


def sample_condition_masks(
    n_parameters: int,
    n_data: int,
    batch_size: int,
    *,
    modes: Sequence[str] = DEFAULT_MODES,
    p_random_low: float = P_RANDOM_LOW,
    p_random_high: float = P_RANDOM_HIGH,
    probs: Optional[Sequence[float]] = None,
    group_ids: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Functional one-shot helper returning ``(batch_size, d)`` condition masks."""
    sampler = ConditionMaskSampler(
        n_parameters,
        n_data,
        modes=modes,
        p_random_low=p_random_low,
        p_random_high=p_random_high,
        probs=probs,
        group_ids=group_ids,
        seed=seed,
    )
    return sampler.sample_numpy(batch_size)


__all__ = [
    "JOINT",
    "POSTERIOR",
    "LIKELIHOOD",
    "RANDOM_LOW",
    "RANDOM_HIGH",
    "DEFAULT_MODES",
    "P_RANDOM_LOW",
    "P_RANDOM_HIGH",
    "normalize_mode",
    "joint_mask",
    "posterior_condition_mask",
    "likelihood_condition_mask",
    "random_condition_mask",
    "mode_mask",
    "group_broadcast",
    "group_collapse",
    "apply_condition",
    "partially_noise",
    "loss_mask",
    "masked_residual",
    "score_matching_loss",
    "conditioned_values",
    "condition_indices",
    "latent_indices",
    "merge_condition_masks",
    "validate_condition_mask",
    "ConditionMaskConfig",
    "ConditionMaskSampler",
    "sample_condition_masks",
]
