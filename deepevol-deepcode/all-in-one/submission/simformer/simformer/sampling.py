"""Arbitrary-conditional sampling with a trained Simformer.

Implements the sampling procedure described in Sec. 3.3 of the paper:

    "We draw samples from the noise distribution and run the reverse diffusion
     process on all unobserved variables, while keeping observed variables
     constant at their conditioning value (Weilbach et al., 2023). Having access
     to all conditional distributions also allows us to combine scores and
     thereby perform inference for simulators with i.i.d. datapoints (Geffner et
     al., 2023). Similarly, we can use other score transformations to adapt to
     other prior or likelihood configurations post-hoc."

and the post-hoc score transformations of Appendix A1.3:

    grad_theta log p_t(theta_t | x_t)
        = grad_theta log p_t(theta_t) + grad_theta log p_t(x_t | theta_t)
    grad_theta log p_t(x_t | theta_t) ~= s_phi(theta_t, t | x_t) - s_phi(theta_t, t)
    grad log p^{a1,b1,a2,b2}(theta_t | x_t)
        ~= a1 * (s(theta_t, t) + b1) + a2 * (s(theta_t, t | x_t) - s(theta_t, t) + b2)

The module is deliberately organised around a single generic reverse-time
integrator (:class:`ConditionalSampler`) onto which any score transformation
(post-hoc prior/likelihood changes, i.i.d. score combination, diffusion
guidance from :mod:`simformer.guidance`) can be plugged.

All heavy lifting (forward noising, marginal statistics, reverse-time Euler-
Maruyama steps) is delegated to :mod:`simformer.diffusion`; the score network
(:mod:`simformer.transformer`) is evaluated under ``torch.no_grad`` while the
integrator itself runs in NumPy so that it can be used from evaluation code
without device/autograd bookkeeping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .condition_masks import (
    DEFAULT_MODES,
    JOINT,
    LIKELIHOOD,
    POSTERIOR,
    ConditionMaskSampler,
    apply_condition,
    condition_indices,
    joint_mask,
    latent_indices,
    likelihood_condition_mask,
    posterior_condition_mask,
)
from .diffusion import (
    MIN_RECOMMENDED_STEPS,
    SDE,
    get_sde,
    reverse_sde_step,
    time_grid,
)

__all__ = [
    "ConditionalSampler",
    "SamplingConfig",
    "SamplingResult",
    "make_score_fn",
    "sample_conditional",
    "sample_posterior",
    "sample_likelihood",
    "sample_joint",
    "sample_parameter_conditional",
    "sample_arbitrary_conditionals",
    "combine_iid_scores",
    "adapt_scores",
    "prior_score_fn",
    "likelihood_score_fn",
    "expand_condition_mask",
    "collapse_condition_mask",
    "random_conditional_targets",
    "DEFAULT_SAMPLING_STEPS",
]

DEFAULT_SAMPLING_STEPS = 500

ArrayLike = Union[np.ndarray, Any]


# ---------------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------------


def _tokenizer_n_values(tokenizer_or_spec: Any) -> Optional[int]:
    """Number of *values* (joint-vector entries) described by a tokenizer/spec."""
    if tokenizer_or_spec is None:
        return None
    for attr in ("input_dim", "n_values", "joint_dim", "dim"):
        if hasattr(tokenizer_or_spec, attr):
            try:
                return int(getattr(tokenizer_or_spec, attr))
            except Exception:  # pragma: no cover - defensive
                pass
    return None


def _tokenizer_n_variables(tokenizer_or_spec: Any) -> Optional[int]:
    """Number of *statistical variables* described by a tokenizer/spec."""
    if tokenizer_or_spec is None:
        return None
    if hasattr(tokenizer_or_spec, "n_variables"):
        try:
            return int(tokenizer_or_spec.n_variables)
        except Exception:  # pragma: no cover - defensive
            pass
    spec = getattr(tokenizer_or_spec, "spec", None)
    if spec is not None:
        n = 0
        n += len(getattr(spec, "parameter_names", ()) or ())
        n += len(getattr(spec, "data_names", ()) or ())
        n += len(getattr(spec, "function_valued", ()) or ())
        if n:
            return n
    n_par = getattr(tokenizer_or_spec, "n_parameter_variables", None)
    n_dat = getattr(tokenizer_or_spec, "n_data_variables", None)
    n_fun = getattr(tokenizer_or_spec, "n_function_tokens", 0) or 0
    if n_par is not None and n_dat is not None:
        return int(n_par) + int(n_dat) + int(n_fun)
    return None


def expand_condition_mask(condition_mask: np.ndarray, tokenizer_or_spec: Any = None) -> np.ndarray:
    """Expand a variable-level mask to joint-vector (value) resolution.

    The tokenizer concatenates ``[theta scalars | x scalars | function-valued
    values]``; the expansion repeats each statistical variable's flag as often
    as it contributes entries to the joint vector.
    """
    mask = np.asarray(condition_mask, dtype=np.float32)
    if mask.ndim == 1:
        mask = mask[None, :]
    n_values = _tokenizer_n_values(tokenizer_or_spec)
    if n_values is None or mask.shape[-1] == n_values:
        return mask
    try:  # reuse the canonical implementation from training.py
        from .training import expand_variable_mask

        return np.asarray(expand_variable_mask(mask, tokenizer_or_spec), dtype=np.float32)
    except Exception:  # pragma: no cover - fallback for standalone use
        pass
    widths = _variable_widths(tokenizer_or_spec)
    if widths is None or int(np.sum(widths)) != n_values:
        raise ValueError(
            f"cannot expand condition mask of width {mask.shape[-1]} to {n_values} values"
        )
    if mask.shape[-1] != len(widths):
        raise ValueError(
            f"condition mask width {mask.shape[-1]} does not match {len(widths)} variables"
        )
    return np.repeat(mask, widths, axis=-1)


def collapse_condition_mask(value_mask: np.ndarray, tokenizer_or_spec: Any = None) -> np.ndarray:
    """Reduce a value-level mask to variable resolution (a variable is observed
    only if all of its entries are observed)."""
    mask = np.asarray(value_mask, dtype=np.float32)
    if mask.ndim == 1:
        mask = mask[None, :]
    n_variables = _tokenizer_n_variables(tokenizer_or_spec)
    if n_variables is None or mask.shape[-1] == n_variables:
        return mask
    widths = _variable_widths(tokenizer_or_spec)
    if widths is None or int(np.sum(widths)) != mask.shape[-1]:
        raise ValueError(
            f"cannot collapse condition mask of width {mask.shape[-1]} to {n_variables} variables"
        )
    out = np.zeros(mask.shape[:-1] + (len(widths),), dtype=np.float32)
    start = 0
    for i, w in enumerate(widths):
        out[..., i] = mask[..., start : start + w].min(axis=-1)
        start += w
    return out


def _variable_widths(tokenizer_or_spec: Any) -> Optional[np.ndarray]:
    if tokenizer_or_spec is None:
        return None
    try:
        from .training import variable_expansion_widths

        return np.asarray(variable_expansion_widths(tokenizer_or_spec), dtype=np.int64)
    except Exception:  # pragma: no cover - defensive
        pass
    spec = getattr(tokenizer_or_spec, "spec", None)
    if spec is None:
        return None
    widths: List[int] = [1] * (len(spec.parameter_names) + len(spec.data_names))
    for fn in spec.function_valued:
        idx = np.asarray(fn.index_set).reshape(-1)
        widths.append(int(idx.shape[0]) if fn.n_index_points is None else int(fn.n_index_points))
    return np.asarray(widths, dtype=np.int64)


def _as_2d_mask(mask: ArrayLike, batch_size: int, dim: int) -> np.ndarray:
    m = np.asarray(mask, dtype=np.float32)
    if m.ndim == 1:
        m = np.repeat(m[None, :], batch_size, axis=0)
    elif m.ndim == 2 and m.shape[0] == 1 and batch_size > 1:
        m = np.repeat(m, batch_size, axis=0)
    if m.shape[-1] != dim:
        raise ValueError(f"condition mask width {m.shape[-1]} does not match joint dim {dim}")
    return m


# ---------------------------------------------------------------------------
# Score-network plumbing
# ---------------------------------------------------------------------------


def _broadcast_t(t: ArrayLike, batch_size: int) -> np.ndarray:
    arr = np.asarray(t, dtype=np.float32).reshape(-1)
    if arr.size == 1:
        arr = np.repeat(arr, batch_size)
    elif arr.size != batch_size:
        raise ValueError(f"t has {arr.size} entries but batch size is {batch_size}")
    return arr


def _to_torch(x: Any, device=None, dtype=None):
    import torch

    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.as_tensor(np.asarray(x, dtype=np.float32))
    if device is not None:
        t = t.to(device)
    if dtype is not None:
        t = t.to(dtype)
    return t


def _flatten_model_output(pred: Any, batch_size: int, dim: int) -> np.ndarray:
    """Coerce a score-network output to ``(batch_size, dim)``."""
    import torch

    if isinstance(pred, (tuple, list)):
        pred = pred[0]
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    arr = np.asarray(pred, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[-1] != dim:
        if arr.size % (batch_size * dim) == 0:
            arr = arr.reshape(batch_size, -1)[:, :dim]
        else:
            raise ValueError(f"score network output {arr.shape} incompatible with {(batch_size, dim)}")
    return arr


def _call_model(
    model: Any,
    x: Any,
    t: Any,
    condition_mask: Any = None,
    attention_mask: Any = None,
    function_values: Any = None,
):
    """Call the score network with a tolerant signature search."""
    attempts = [
        dict(condition_mask=condition_mask, attention_mask=attention_mask, function_values=function_values),
        dict(condition_mask=condition_mask, attention_mask=attention_mask),
        dict(condition_mask=condition_mask),
        dict(),
    ]
    last_err: Optional[Exception] = None
    for kwargs in attempts:
        try:
            return model(x, t, **kwargs)
        except TypeError as err:  # unexpected keyword -> try a simpler call
            last_err = err
            continue
        except Exception as err:  # unexpected kwarg wrapped differently
            msg = str(err)
            if "unexpected keyword" in msg or "got multiple values" in msg:
                last_err = err
                continue
            raise
    if last_err is not None:  # pragma: no cover - defensive
        raise last_err
    raise RuntimeError("could not call the score network")


def make_score_fn(
    model: Any,
    sde: Optional[SDE] = None,
    *,
    condition_mask: Optional[ArrayLike] = None,
    condition_values: Optional[ArrayLike] = None,
    attention_mask: Any = None,
    attention_mask_fn: Optional[Callable[[np.ndarray], Any]] = None,
    function_values: Any = None,
    score_transform: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = None,
    dim: Optional[int] = None,
    device: Any = None,
    dtype: Any = None,
    batch_chunk: Optional[int] = None,
) -> Callable[[np.ndarray, ArrayLike], np.ndarray]:
    """Build a NumPy score function for the diffusion of ``(theta, x)``.

    Parameters
    ----------
    model:
        Trained score network (``SimformerScoreNetwork``). Predicts epsilon, so
        ``grad log p_t = -eps / sigma(t)``.
    sde:
        Forward SDE; defaults to VESDE.
    condition_mask / condition_values:
        Observed variables are clamped to ``condition_values`` before the score
        is evaluated, exactly as defined by
        ``x_hat_t^{M_C} = (1 - M_C) x_hat_t + M_C x_hat_0``.
    attention_mask / attention_mask_fn:
        Fixed attention mask, or a callable mapping ``M_C -> M_E`` (used to plug
        in the graph-inversion-based dynamic masks).
    score_transform:
        Optional ``fn(score, x_hat_t, t) -> score`` hook used by guidance and by
        the post-hoc prior/likelihood adaptations of App. A1.3.
    """
    sde = sde if sde is not None else get_sde("vesde")

    def score_fn(x: np.ndarray, t: ArrayLike) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim == 1:
            x_arr = x_arr[None, :]
        b, d = x_arr.shape
        if dim is not None and d != dim:
            raise ValueError(f"expected joint dim {dim}, got {d}")
        t_arr = _broadcast_t(t, b)
        mask_arr = None
        if condition_mask is not None:
            mask_arr = _as_2d_mask(condition_mask, b, d)
            if condition_values is not None:
                cv = np.asarray(condition_values, dtype=np.float32)
                if cv.ndim == 1:
                    cv = np.broadcast_to(cv[None, :], (b, d))
                x_in = (1.0 - mask_arr) * x_arr + mask_arr * cv
            else:
                x_in = x_arr
        else:
            x_in = x_arr

        attn = attention_mask
        if isinstance(attn, np.ndarray) and attn.ndim == 3 and attn.shape[0] == b:
            pass  # already per-batch
        elif callable(attention_mask_fn):
            attn = attention_mask_fn(mask_arr if mask_arr is not None else np.zeros((b, d), np.float32))
        elif attention_mask_fn is not None and attn is None:
            attn = attention_mask_fn

        chunks: List[np.ndarray] = []
        step = batch_chunk or b
        for start in range(0, b, step):
            stop = min(start + step, b)
            xb = _to_torch(x_in[start:stop], device=device, dtype=dtype)
            tb = _to_torch(t_arr[start:stop], device=device, dtype=dtype)
            cmb = _to_torch(mask_arr[start:stop], device=device, dtype=dtype) if mask_arr is not None else None
            fvb = None
            if function_values is not None:
                fv = np.asarray(function_values, dtype=np.float32)
                fvb = _to_torch(fv[start:stop] if fv.ndim > 0 and fv.shape[0] == b else fv, device=device, dtype=dtype)
            pred = _call_model(model, xb, tb, condition_mask=cmb, attention_mask=attn, function_values=fvb)
            eps = _flatten_model_output(pred, stop - start, d)
            chunks.append(eps)
        eps_all = np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]

        sigma = np.asarray(sde.marginal_std(t_arr), dtype=np.float32).reshape(b, -1)
        if sigma.shape[-1] == 1:
            sigma = np.repeat(sigma, d, axis=-1)
        score = -eps_all / np.maximum(sigma, 1e-12)
        if score_transform is not None:
            score = score_transform(score, x_in, t_arr)
        return np.asarray(score, dtype=np.float32)

    return score_fn


# ---------------------------------------------------------------------------
# Score transformations (App. A1.3)
# ---------------------------------------------------------------------------


def combine_iid_scores(
    base_score_fn: Callable[[np.ndarray, ArrayLike], np.ndarray],
    conditional_score_fns: Sequence[Callable[[np.ndarray, ArrayLike], np.ndarray]],
) -> Callable[[np.ndarray, ArrayLike], np.ndarray]:
    """Combine scores for i.i.d. data (Geffner et al., 2023).

    Using ``grad log p(theta | x_1..x_N) = grad log p(theta) + sum_i grad log
    p(x_i | theta)`` and the approximation

        ``grad log p_t(x_i | theta) ~= s(theta, t | x_i) - s(theta, t)``,

    the combined score is ``s(theta, t) + sum_i (s(theta, t | x_i) - s(theta, t))``.
    """

    def combined(x: np.ndarray, t: ArrayLike) -> np.ndarray:
        base = np.asarray(base_score_fn(x, t), dtype=np.float32)
        total = base.copy()
        for fn in conditional_score_fns:
            total = total + (np.asarray(fn(x, t), dtype=np.float32) - base)
        return total

    return combined


def adapt_scores(
    joint_score_fn: Callable[[np.ndarray, ArrayLike], np.ndarray],
    conditional_score_fn: Optional[Callable[[np.ndarray, ArrayLike], np.ndarray]] = None,
    *,
    alpha_prior: float = 1.0,
    beta_prior: float = 0.0,
    alpha_likelihood: float = 1.0,
    beta_likelihood: float = 0.0,
) -> Callable[[np.ndarray, ArrayLike], np.ndarray]:
    """Post-hoc prior/likelihood adaptation (App. A1.3, Eq. 3).

        a1 * (s(theta_t, t) + b1) + a2 * (s(theta_t, t | x_t) - s(theta_t, t) + b2)

    With ``conditional_score_fn=None`` only the prior term (tempering/shifting
    the prior) is applied.
    """

    def adapted(x: np.ndarray, t: ArrayLike) -> np.ndarray:
        prior = np.asarray(joint_score_fn(x, t), dtype=np.float32)
        out = alpha_prior * (prior + beta_prior)
        if conditional_score_fn is not None:
            cond = np.asarray(conditional_score_fn(x, t), dtype=np.float32)
            out = out + alpha_likelihood * (cond - prior + beta_likelihood)
        return out

    return adapted


def prior_score_fn(score_fn: Callable[[np.ndarray, ArrayLike], np.ndarray]) -> Callable:
    """Identity helper documenting ``s_phi(theta_t, t)`` (the joint/unconditional
    score) used in the decompositions of App. A1.3."""
    return score_fn


def likelihood_score_fn(
    joint_score_fn: Callable[[np.ndarray, ArrayLike], np.ndarray],
    conditional_score_fn: Callable[[np.ndarray, ArrayLike], np.ndarray],
) -> Callable[[np.ndarray, ArrayLike], np.ndarray]:
    """Approximate likelihood score ``s(theta_t, t | x_t) - s(theta_t, t)``."""

    def fn(x: np.ndarray, t: ArrayLike) -> np.ndarray:
        return np.asarray(conditional_score_fn(x, t), dtype=np.float32) - np.asarray(
            joint_score_fn(x, t), dtype=np.float32
        )

    return fn


# ---------------------------------------------------------------------------
# Configuration / result containers
# ---------------------------------------------------------------------------


@dataclass
class SamplingConfig:
    """Numerical settings of the reverse diffusion sampler."""

    n_steps: int = DEFAULT_SAMPLING_STEPS
    t_min: float = 1e-5
    t_max: float = 1.0
    n_samples: int = 1000
    batch_size: Optional[int] = None
    seed: Optional[int] = None
    return_trajectory: bool = False
    clamp_every_step: bool = True
    self_recurrence: int = 1
    device: Any = None
    dtype: Any = None

    def time_grid(self, sde: Optional[SDE] = None) -> np.ndarray:
        t_min = self.t_min if sde is None else getattr(sde, "t_min", self.t_min)
        t_max = self.t_max if sde is None else getattr(sde, "t_max", self.t_max)
        return time_grid(self.n_steps, t_min, t_max, descending=True)


@dataclass
class SamplingResult:
    """Output of a conditional sampling call."""

    samples: np.ndarray  # (n_samples, dim)
    condition_mask: Optional[np.ndarray] = None
    condition_values: Optional[np.ndarray] = None
    trajectory: Optional[np.ndarray] = None
    times: Optional[np.ndarray] = None
    mode: Optional[str] = None

    def __len__(self) -> int:  # pragma: no cover - convenience
        return int(self.samples.shape[0])

    def as_array(self) -> np.ndarray:  # pragma: no cover - convenience
        return self.samples


# ---------------------------------------------------------------------------
# The sampler
# ---------------------------------------------------------------------------


class ConditionalSampler:
    """Sample arbitrary conditionals of a trained Simformer (Sec. 3.3).

    A single trained score network represents the joint ``p(theta, x)``; running
    the reverse SDE on the latent coordinates while clamping the observed ones
    yields the corresponding conditional. The class offers shorthands for the
    posterior ``p(theta | x)``, the likelihood ``p(x | theta)``, the joint
    ``p(theta, x)`` and fully arbitrary parameter/data conditionals, plus hooks
    for score transformations (i.i.d. combination, post-hoc prior/likelihood
    changes, interval guidance).
    """

    def __init__(
        self,
        model: Any,
        sde: Optional[SDE] = None,
        tokenizer: Any = None,
        attention_mask: Any = None,
        attention_mask_fn: Optional[Callable[[np.ndarray], Any]] = None,
        *,
        config: Optional[SamplingConfig] = None,
        condition_sampler: Optional[ConditionMaskSampler] = None,
        device: Any = None,
        dtype: Any = None,
        dim: Optional[int] = None,
        n_parameters: Optional[int] = None,
        n_data: Optional[int] = None,
    ) -> None:
        self.model = model
        self.sde = sde if sde is not None else get_sde("vesde")
        self.tokenizer = tokenizer if tokenizer is not None else getattr(model, "tokenizer", None)
        self.attention_mask = attention_mask
        self.attention_mask_fn = attention_mask_fn
        self.config = config or SamplingConfig()
        self.device = self.config.device if device is None else device
        self.dtype = self.config.dtype if dtype is None else dtype
        self.dim = int(dim) if dim is not None else self._infer_dim()
        self.n_parameters = n_parameters if n_parameters is not None else getattr(self.tokenizer, "n_parameter_variables", None)
        self.n_data = n_data if n_data is not None else getattr(self.tokenizer, "n_data_variables", None)
        self._rng = np.random.default_rng(self.config.seed)
        self.condition_sampler = condition_sampler or ConditionMaskSampler(
            self.n_parameters or max(self.dim - 1, 1),
            self.n_data or 1,
            modes=DEFAULT_MODES,
            seed=self.config.seed,
        )

    # -- introspection ----------------------------------------------------
    def _infer_dim(self) -> int:
        for source in (self.tokenizer, self.model):
            if source is None:
                continue
            for attr in ("input_dim", "n_values", "joint_dim"):
                if hasattr(source, attr):
                    try:
                        val = getattr(source, attr)
                        if val is not None:
                            return int(val)
                    except Exception:  # pragma: no cover - defensive
                        pass
        raise ValueError("could not infer the joint dimension; pass dim=...")

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    @property
    def n_variables(self) -> Optional[int]:
        n = _tokenizer_n_variables(self.tokenizer)
        return n if n is not None else self.n_parameters and (self.n_parameters + self.n_data + 0)

    # -- mask helpers -----------------------------------------------------
    def value_mask(self, mask: ArrayLike) -> np.ndarray:
        """Coerce a variable- or value-level mask to value (joint) resolution."""
        m = np.asarray(mask, dtype=np.float32)
        if m.ndim == 1:
            m = m[None, :]
        if m.shape[-1] == self.dim:
            return m
        return expand_condition_mask(m, self.tokenizer)

    def variable_mask(self, mask: ArrayLike) -> np.ndarray:
        m = np.asarray(mask, dtype=np.float32)
        if m.ndim == 1:
            m = m[None, :]
        if m.shape[-1] == self.dim:
            return collapse_condition_mask(m, self.tokenizer)
        return m

    def posterior_condition_mask(self) -> np.ndarray:
        """M_C for ``p(theta | x)``: parameters latent, data observed."""
        if self.n_parameters is not None and self.n_data is not None:
            base = posterior_condition_mask(self.n_parameters, self.n_data)
            return self.value_mask(base)
        raise ValueError("n_parameters/n_data unknown; set them on the sampler")

    def likelihood_condition_mask(self) -> np.ndarray:
        """M_C for ``p(x | theta)``: parameters observed, data latent."""
        if self.n_parameters is not None and self.n_data is not None:
            base = likelihood_condition_mask(self.n_parameters, self.n_data)
            return self.value_mask(base)
        raise ValueError("n_parameters/n_data unknown; set them on the sampler")

    def joint_condition_mask(self) -> np.ndarray:
        return self.value_mask(joint_mask(self.dim))

    # -- score functions --------------------------------------------------
    def score_fn(
        self,
        condition_mask: Optional[ArrayLike] = None,
        condition_values: Optional[ArrayLike] = None,
        *,
        attention_mask: Any = "__default__",
        score_transform: Optional[Callable] = None,
        function_values: Any = None,
    ) -> Callable[[np.ndarray, ArrayLike], np.ndarray]:
        attn = self.attention_mask if attention_mask == "__default__" else attention_mask
        return make_score_fn(
            self.model,
            self.sde,
            condition_mask=condition_mask,
            condition_values=condition_values,
            attention_mask=attn,
            attention_mask_fn=self.attention_mask_fn,
            function_values=function_values,
            score_transform=score_transform,
            dim=self.dim,
            device=self.device,
            dtype=self.dtype,
        )

    def _init_samples(self, n_samples: int, batch_size: Optional[int] = None) -> np.ndarray:
        if batch_size is None or batch_size >= n_samples:
            return self._draw_prior(n_samples)
        parts = []
        remaining = n_samples
        while remaining > 0:
            b = min(batch_size, remaining)
            parts.append(self._draw_prior(b))
            remaining -= b
        return np.concatenate(parts, axis=0)

    def _draw_prior(self, n_samples: int) -> np.ndarray:
        try:
            x = self.sde.sample_prior((n_samples, self.dim), rng=self._rng)
        except TypeError:  # pragma: no cover - older signature
            x = self.sde.sample_prior((n_samples, self.dim), self._rng)
        return np.asarray(x, dtype=np.float32)

    # -- the integrator ---------------------------------------------------
    def _reverse_sde(
        self,
        score_fn: Callable[[np.ndarray, ArrayLike], np.ndarray],
        x: np.ndarray,
        *,
        condition_mask: Optional[np.ndarray] = None,
        condition_values: Optional[np.ndarray] = None,
        n_steps: Optional[int] = None,
        return_trajectory: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        sde = self.sde
        n_steps = int(n_steps or self.config.n_steps)
        t_grid = time_grid(
            n_steps,
            getattr(sde, "t_min", self.config.t_min),
            getattr(sde, "t_max", self.config.t_max),
            descending=True,
        )
        if condition_mask is not None:
            condition_mask = _as_2d_mask(condition_mask, x.shape[0], self.dim)
            if condition_values is not None:
                cv = np.asarray(condition_values, dtype=np.float32)
                if cv.ndim == 1:
                    cv = np.broadcast_to(cv[None, :], x.shape)
                x = (1.0 - condition_mask) * x + condition_mask * cv

        traj = [x.copy()] if return_trajectory else None
        for i in range(len(t_grid) - 1):
            t_cur = float(t_grid[i])
            t_next = float(t_grid[i + 1])
            if condition_mask is not None and self.config.clamp_every_step and condition_values is not None:
                cv = np.asarray(condition_values, dtype=np.float32)
                if cv.ndim == 1:
                    cv = np.broadcast_to(cv[None, :], x.shape)
                x = (1.0 - condition_mask) * x + condition_mask * cv
            for _ in range(max(1, int(self.config.self_recurrence))):
                score = score_fn(x, np.full(x.shape[0], t_cur, dtype=np.float32))
                noise = self._rng.standard_normal(x.shape).astype(np.float32)
                x = self._em_step(x, t_cur, t_next, score, noise)
            if condition_mask is not None and condition_values is not None:
                cv = np.asarray(condition_values, dtype=np.float32)
                if cv.ndim == 1:
                    cv = np.broadcast_to(cv[None, :], x.shape)
                x = (1.0 - condition_mask) * x + condition_mask * cv
            if return_trajectory:
                traj.append(x.copy())
        return x, (np.stack(traj, axis=0) if traj is not None else None), t_grid

    def _em_step(
        self,
        x: np.ndarray,
        t_cur: float,
        t_next: float,
        score: np.ndarray,
        noise: np.ndarray,
    ) -> np.ndarray:
        """One reverse-time Euler-Maruyama step (delegates to diffusion.py)."""
        try:
            out = reverse_sde_step(self.sde, x, t_cur, t_next, score, noise=noise)
            return np.asarray(out, dtype=np.float32)
        except TypeError:
            try:
                out = reverse_sde_step(self.sde, x, t_cur, t_next, score, noise)
                return np.asarray(out, dtype=np.float32)
            except TypeError:
                pass
        # Fallback: explicit Anderson reverse SDE discretisation.
        dt = t_next - t_cur  # negative
        g = float(np.asarray(self.sde.diffusion(t_cur)).reshape(-1)[0])
        f = np.asarray(self.sde.drift(x, t_cur), dtype=np.float32)
        return (
            x
            + (f - (g ** 2) * score) * dt
            + g * math.sqrt(max(-dt, 0.0)) * noise
        ).astype(np.float32)

    # -- public sampling API ---------------------------------------------
    def sample(
        self,
        condition_mask: Optional[ArrayLike] = None,
        condition_values: Optional[ArrayLike] = None,
        *,
        n_samples: Optional[int] = None,
        n_steps: Optional[int] = None,
        attention_mask: Any = "__default__",
        score_transform: Optional[Callable] = None,
        function_values: Any = None,
        return_trajectory: bool = False,
        mode: Optional[str] = None,
        x_init: Optional[np.ndarray] = None,
    ) -> SamplingResult:
        """Sample from ``p(latent | observed)`` with the reverse SDE."""
        n_samples = int(n_samples or self.config.n_samples)
        mask_values = self.value_mask(condition_mask).astype(np.float32) if condition_mask is not None else None
        if mask_values is not None and mask_values.shape[0] == 1:
            mask_values = np.repeat(mask_values, n_samples, axis=0)
        cv = None
        if condition_values is not None:
            cv = np.asarray(condition_values, dtype=np.float32)
            if cv.ndim == 1 and mask_values is not None and cv.shape[0] == mask_values.shape[-1]:
                cv = np.repeat(cv[None, :], n_samples, axis=0)
        score_fn = self.score_fn(
            mask_values,
            cv,
            attention_mask=attention_mask,
            score_transform=score_transform,
            function_values=function_values,
        )
        if x_init is None:
            x = self._init_samples(n_samples, self.config.batch_size)
        else:
            x = np.asarray(x_init, dtype=np.float32).copy()
        x, traj, times = self._reverse_sde(
            score_fn,
            x,
            condition_mask=mask_values,
            condition_values=cv,
            n_steps=n_steps,
            return_trajectory=return_trajectory or self.config.return_trajectory,
        )
        return SamplingResult(
            samples=x,
            condition_mask=None if mask_values is None else mask_values[:1],
            condition_values=None if cv is None else cv[:1],
            trajectory=traj,
            times=times,
            mode=mode,
        )

    # Convenience wrappers -------------------------------------------------
    def posterior(
        self,
        x_obs: ArrayLike,
        *,
        n_samples: Optional[int] = None,
        n_steps: Optional[int] = None,
        **kwargs: Any,
    ) -> SamplingResult:
        """``p(theta | x)`` — data observed, parameters inferred."""
        mask = self.posterior_condition_mask()
        return self.sample(mask, x_obs, n_samples=n_samples, n_steps=n_steps, mode=POSTERIOR, **kwargs)

    def likelihood(
        self,
        theta: ArrayLike,
        *,
        n_samples: Optional[int] = None,
        n_steps: Optional[int] = None,
        **kwargs: Any,
    ) -> SamplingResult:
        """``p(x | theta)`` — parameters observed, data sampled."""
        mask = self.likelihood_condition_mask()
        return self.sample(mask, theta, n_samples=n_samples, n_steps=n_steps, mode=LIKELIHOOD, **kwargs)

    def joint(
        self,
        *,
        n_samples: Optional[int] = None,
        n_steps: Optional[int] = None,
        **kwargs: Any,
    ) -> SamplingResult:
        """``p(theta, x)`` — nothing conditioned."""
        mask = self.joint_condition_mask()
        return self.sample(mask, None, n_samples=n_samples, n_steps=n_steps, mode=JOINT, **kwargs)

    def parameter_conditional(
        self,
        observed: ArrayLike,
        observed_mask: ArrayLike,
        *,
        n_samples: Optional[int] = None,
        n_steps: Optional[int] = None,
        **kwargs: Any,
    ) -> SamplingResult:
        """Fully general conditional: observed entries given by ``observed_mask``,
        values given by ``observed`` (zeros are ignored where mask is 0)."""
        mask = self.value_mask(observed_mask)
        values = np.asarray(observed, dtype=np.float32)
        if values.ndim == 1:
            values = values[None, :]
        values = values * mask
        return self.sample(mask, values, n_samples=n_samples, n_steps=n_steps, **kwargs)

    def sample_from_condition_mask(
        self,
        condition_mask: ArrayLike,
        condition_values: ArrayLike,
        *,
        n_samples: Optional[int] = None,
        n_steps: Optional[int] = None,
        **kwargs: Any,
    ) -> SamplingResult:
        return self.sample(condition_mask, condition_values, n_samples=n_samples, n_steps=n_steps, **kwargs)

    # -- reference-free utilities ------------------------------------------
    def score_estimate(
        self,
        x: ArrayLike,
        t: float = 1e-3,
        condition_mask: Optional[ArrayLike] = None,
        *,
        return_epsilon: bool = False,
    ) -> np.ndarray:
        """Evaluate the trained score (or epsilon) at a given noise level."""
        fn = self.score_fn(condition_mask)
        s = fn(np.asarray(x, dtype=np.float32), np.full(len(np.atleast_2d(x)), t, dtype=np.float32))
        if return_epsilon:
            sigma = np.asarray(self.sde.marginal_std(np.full(s.shape[0], t, dtype=np.float32)), dtype=np.float32)
            return -s * sigma.reshape(-1, 1)
        return s


# ---------------------------------------------------------------------------
# Functional entry points
# ---------------------------------------------------------------------------


def sample_conditional(
    model: Any,
    condition_mask: ArrayLike,
    condition_values: ArrayLike,
    *,
    sde: Optional[SDE] = None,
    tokenizer: Any = None,
    attention_mask: Any = None,
    attention_mask_fn: Optional[Callable] = None,
    n_samples: int = 1000,
    n_steps: int = DEFAULT_SAMPLING_STEPS,
    t_min: float = 1e-5,
    t_max: float = 1.0,
    seed: Optional[int] = None,
    device: Any = None,
    dtype: Any = None,
    score_transform: Optional[Callable] = None,
    return_trajectory: bool = False,
    return_result: bool = False,
    **kwargs: Any,
) -> Union[np.ndarray, SamplingResult]:
    """Sample an arbitrary conditional ``p(latent | observed)`` of a Simformer.

    Runs the reverse diffusion process on all unobserved variables while keeping
    observed variables constant at their conditioning value (Sec. 3.3).
    """
    sampler = ConditionalSampler(
        model,
        sde=sde,
        tokenizer=tokenizer,
        attention_mask=attention_mask,
        attention_mask_fn=attention_mask_fn,
        config=SamplingConfig(
            n_steps=n_steps,
            t_min=t_min,
            t_max=t_max,
            n_samples=n_samples,
            seed=seed,
            return_trajectory=return_trajectory,
            device=device,
            dtype=dtype,
        ),
        device=device,
        dtype=dtype,
    )
    result = sampler.sample(
        condition_mask,
        condition_values,
        n_samples=n_samples,
        n_steps=n_steps,
        score_transform=score_transform,
        return_trajectory=return_trajectory,
        **kwargs,
    )
    return result if return_result else result.samples


def sample_posterior(
    model: Any,
    x_obs: ArrayLike,
    *,
    sde: Optional[SDE] = None,
    tokenizer: Any = None,
    n_samples: int = 1000,
    n_steps: int = DEFAULT_SAMPLING_STEPS,
    condition_values_full: Optional[ArrayLike] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Sample ``p(theta | x)`` (posterior) for one or more observations."""
    sampler_kwargs = dict(sde=sde, tokenizer=tokenizer, n_samples=n_samples, n_steps=n_steps)
    sampler_kwargs.update({k: kwargs.pop(k) for k in ("attention_mask", "attention_mask_fn", "device", "dtype") if k in kwargs})
    sampler = ConditionalSampler(model, config=SamplingConfig(n_steps=n_steps, n_samples=n_samples, seed=kwargs.pop("seed", None)), **sampler_kwargs)
    values = x_obs if condition_values_full is None else condition_values_full
    return sampler.posterior(values, n_samples=n_samples, n_steps=n_steps, **kwargs).samples


def sample_likelihood(
    model: Any,
    theta: ArrayLike,
    *,
    sde: Optional[SDE] = None,
    tokenizer: Any = None,
    n_samples: int = 1000,
    n_steps: int = DEFAULT_SAMPLING_STEPS,
    **kwargs: Any,
) -> np.ndarray:
    """Sample ``p(x | theta)`` (likelihood / simulator emulation)."""
    sampler = ConditionalSampler(
        model,
        sde=sde,
        tokenizer=tokenizer,
        attention_mask=kwargs.pop("attention_mask", None),
        attention_mask_fn=kwargs.pop("attention_mask_fn", None),
        config=SamplingConfig(n_steps=n_steps, n_samples=n_samples, seed=kwargs.pop("seed", None)),
    )
    return sampler.likelihood(theta, n_samples=n_samples, n_steps=n_steps, **kwargs).samples


def sample_joint(
    model: Any,
    *,
    sde: Optional[SDE] = None,
    tokenizer: Any = None,
    n_samples: int = 1000,
    n_steps: int = DEFAULT_SAMPLING_STEPS,
    **kwargs: Any,
) -> np.ndarray:
    """Sample from the joint ``p(theta, x)``."""
    sampler = ConditionalSampler(
        model,
        sde=sde,
        tokenizer=tokenizer,
        attention_mask=kwargs.pop("attention_mask", None),
        attention_mask_fn=kwargs.pop("attention_mask_fn", None),
        config=SamplingConfig(n_steps=n_steps, n_samples=n_samples, seed=kwargs.pop("seed", None)),
    )
    return sampler.joint(n_samples=n_samples, n_steps=n_steps, **kwargs).samples


def sample_parameter_conditional(
    model: Any,
    joint: ArrayLike,
    observed_mask: ArrayLike,
    *,
    sde: Optional[SDE] = None,
    tokenizer: Any = None,
    n_samples: int = 1000,
    n_steps: int = DEFAULT_SAMPLING_STEPS,
    **kwargs: Any,
) -> np.ndarray:
    """Sample an arbitrary parameter/data conditional given a partially observed
    joint vector."""
    sampler = ConditionalSampler(
        model,
        sde=sde,
        tokenizer=tokenizer,
        attention_mask=kwargs.pop("attention_mask", None),
        attention_mask_fn=kwargs.pop("attention_mask_fn", None),
        config=SamplingConfig(n_steps=n_steps, n_samples=n_samples, seed=kwargs.pop("seed", None)),
    )
    return sampler.parameter_conditional(
        joint, observed_mask, n_samples=n_samples, n_steps=n_steps, **kwargs
    ).samples


# ---------------------------------------------------------------------------
# Task-level helpers
# ---------------------------------------------------------------------------


def random_conditional_targets(
    n_variables: int,
    n_targets: int = 100,
    *,
    seed: Optional[int] = None,
    allow_fully_observed: bool = False,
    allow_empty: bool = True,
    p_random_low: float = 0.3,
    p_random_high: float = 0.7,
) -> np.ndarray:
    """Draw random conditional targets as in Sec. 4.1 ("100 random conditional
    targets"): a mix of joint, posterior, likelihood and Bernoulli masks.

    Returns an ``(n_targets, n_variables)`` array of ``M_C`` masks.
    """
    rng = np.random.default_rng(seed)
    n_parameters = n_variables // 2
    n_data = n_variables - n_parameters
    sampler = ConditionMaskSampler(
        n_parameters,
        n_data,
        modes=DEFAULT_MODES,
        p_random_low=p_random_low,
        p_random_high=p_random_high,
        seed=seed,
    )
    masks = sampler.sample_numpy(n_targets)
    masks = np.asarray(masks, dtype=np.float32)
    if not allow_fully_observed:
        bad = masks.sum(axis=1) == n_variables
        masks[bad, rng.integers(0, n_variables, size=int(bad.sum()))] = 0.0
    if not allow_empty:
        bad = masks.sum(axis=1) == 0
        masks[bad, rng.integers(0, n_variables, size=int(bad.sum()))] = 1.0
    return masks


def sample_arbitrary_conditionals(
    model: Any,
    ground_truth_joint: ArrayLike,
    *,
    n_targets: int = 100,
    n_samples: int = 1000,
    n_steps: int = DEFAULT_SAMPLING_STEPS,
    sde: Optional[SDE] = None,
    tokenizer: Any = None,
    seed: Optional[int] = None,
    return_masks: bool = False,
    **kwargs: Any,
) -> Union[List[np.ndarray], Tuple[List[np.ndarray], np.ndarray]]:
    """Sample ``n_targets`` different conditionals of a trained Simformer.

    ``ground_truth_joint`` supplies the conditioning values (a single joint
    sample, e.g. drawn from the true simulator). Returns one sample array per
    conditional, matching the protocol of Sec. 4.1.
    """
    x0 = np.asarray(ground_truth_joint, dtype=np.float32)
    if x0.ndim == 1:
        x0 = x0[None, :]
    n_values = x0.shape[-1]
    masks = random_conditional_targets(n_values, n_targets, seed=seed)
    sampler = ConditionalSampler(
        model,
        sde=sde,
        tokenizer=tokenizer,
        attention_mask=kwargs.pop("attention_mask", None),
        attention_mask_fn=kwargs.pop("attention_mask_fn", None),
        config=SamplingConfig(n_steps=n_steps, n_samples=n_samples, seed=seed),
    )
    samples: List[np.ndarray] = []
    for i in range(masks.shape[0]):
        values = x0[0] * masks[i]
        res = sampler.sample(
            masks[i : i + 1],
            values[None, :],
            n_samples=n_samples,
            n_steps=n_steps,
            **kwargs,
        )
        samples.append(res.samples)
    if return_masks:
        return samples, masks
    return samples
