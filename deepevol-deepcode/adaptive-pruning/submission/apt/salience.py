"""Outlier-aware salience scoring for APT.

Implements the salience scoring function used by APT to identify the pruning and
tuning parameter blocks during LM fine-tuning (Section 4.2 and Appendix B):

    S(W_{i,j}) = | W_{i,j} * dL / dW_{i,j} |                                    (3)

Because the gradients of the *frozen* weights are unreachable in a PEFT setting,
APT computes the salience as the magnitude of the product between activations
and their gradients, i.e. - for a linear layer ``H = W X`` printed with the
column convention ``W_{:,j}`` - :

    S~_t(W_{:,j}) = [ sum_{x,y} sum_i | dL / dH_{j,i} | ] * [ sum_{x,y} sum_i | H_{j,i} | ]

The activation/gradient tensors are compressed by summing along the batch (and
sequence) dimension *before* the product, which is what makes the score cheap to
compute in memory.  On top of that, the squared-root of the kurtosis of the
"activation" ``O_{:,j} = W_{:,j} o X_{j,:}^T`` is added so that blocks carrying
outlier parameters (which hold task-specific capabilities) are kept longer:

    S^(W_{:,j}) = S~(W_{:,j}) + sqrt( Kurt(O_{j,:}) )

For the APT adapter of Equation (1), ``H_apt(X) = m_o o (W + s W_B W_A) X o m_i``,
the layer salience is the *sum* of the frozen-weight salience and the
corresponding tuning-weight salience (Appendix B):

    S(H, i) = sum_p |dL/dW_{i,p} * W_{i,p}| + s * sum_q |dL/dW_{B i,q} * W_{B i,q}|
    S(H, j) = sum_p |dL/dW_{p,j} * W_{p,j}| + s * sum_q |dL/dW_{A q,j} * W_{A q,j}|
    S(H, k) = s * sum_l |dL/dW_{A k,l} * W_{A k,l}| = s * sum_l |dL/dW_{B l,k} * W_{B l,k}|

so the "input dimension" score is taken from the weight column ``W_{p,j}`` /
``W_A[:,j]`` (equivalently from ``H_{j,:} * dL/dH_{j,:}``, the form actually used
here since gradients of frozen weights are unavailable), the "output dimension"
score from the weight row, and the "tuning rank" score from either ``W_A`` or
``W_B`` (the ``W_B`` variant is the adapter importance
``I(H_apt) = sum_{i,j} S(W_{B i,j})`` of Section 4.3).

The global score follows Algorithm 1 / the Addendum ("APT Implementation"):

    S_bar^(t)(m) <- beta * S_bar^(t-1)(m) + (1 - beta) * S^(m),   beta = 0.85

i.e. ``0.85 * previous + 0.15 * current``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

try:  # block-type constants live with the adapter implementation
    from .adapters import DIMENSION, HEAD, NEURON, iter_masked_linears
except Exception:  # pragma: no cover - defensive fallback for standalone use
    HEAD, NEURON, DIMENSION = 0, 1, 2

    def iter_masked_linears(module, names=None):  # type: ignore
        for name, mod in module.named_modules():
            if hasattr(mod, "base_weight") and hasattr(mod, "adapter"):
                if names is None or name in names:
                    yield name, mod


__all__ = [
    "DEFAULT_BETA",
    "DEFAULT_KURTOSIS_CHUNK",
    "HEAD",
    "NEURON",
    "DIMENSION",
    "SalienceConfig",
    "MovingAverageSalience",
    "OutlierAwareSalience",
    "SalienceTracker",
    "activation_gradient_product",
    "column_kurtosis",
    "kurtosis_term",
    "moving_average_update",
    "module_site",
    "weight_grad_scores",
]

DEFAULT_BETA = 0.85
DEFAULT_KURTOSIS_CHUNK = 2048

# Site tokens used to recover the attention site ("query"/"value"/...) from a
# wrapped module name such as ``roberta.encoder.layer.0.attention.self.query``.
_SITE_ALIASES: Tuple[Tuple[str, str], ...] = (
    ("query", "query"),
    ("q_proj", "query"),
    ("q_lin", "query"),
    ("value", "value"),
    ("v_proj", "value"),
    ("v_lin", "value"),
    ("key", "key"),
    ("k_proj", "key"),
    ("out_proj", "out"),
    ("dense", "out"),
    ("wi_0", "ffn_in"),
    ("wi_1", "ffn_in"),
    ("wi", "ffn_in"),
    ("fc1", "ffn_in"),
    ("intermediate", "ffn_in"),
    ("ffn_in", "ffn_in"),
    ("wi_0", "ffn_in"),
    ("wo", "ffn_out"),
    ("fc2", "ffn_out"),
    ("output", "ffn_out"),
    ("ffn_out", "ffn_out"),
)


def module_site(name: str, kind: int = HEAD) -> str:
    """Best-effort attention/FFN site label for a wrapped module path."""
    lowered = (name or "").lower()
    for token, site in _SITE_ALIASES:
        if token in lowered:
            if kind == HEAD and site in ("ffn_in", "ffn_out"):
                continue
            return site
    if kind == HEAD:
        # A head block always lives in an attention projection.
        for token in ("attention", "attn", "mha"):
            if token in lowered:
                return "query"
    return "ffn" if kind == NEURON else "unknown"


# ---------------------------------------------------------------------------
# numeric helpers
# ---------------------------------------------------------------------------
def moving_average_update(
    previous: Union[float, torch.Tensor],
    current: Union[float, torch.Tensor],
    beta: float = DEFAULT_BETA,
) -> Union[float, torch.Tensor]:
    """``beta * previous + (1 - beta) * current`` (Algorithm 1 / Addendum)."""
    return beta * previous + (1.0 - beta) * current


@torch.no_grad()
def column_kurtosis(
    weight_column: torch.Tensor,
    activation_column: torch.Tensor,
    *,
    chunk: int = DEFAULT_KURTOSIS_CHUNK,
    max_samples: Optional[int] = None,
    fisher: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Kurtosis of ``O_{:,j} = W_{:,j} o X_{j,:}^T`` for one column ``j``.

    ``weight_column`` has shape ``(d_o,)`` and ``activation_column`` shape
    ``(B,)`` (batch/sequence already flattened).  Returns a ``(d_o,)`` tensor
    holding the kurtosis of ``O[i, :] = W[i, j] * X[:, j]``.

    The moments are accumulated in chunks of examples to bound memory
    (a single column of ``O`` would otherwise be ``d_o x (batch * seq)``).
    The kurtosis is the *Pearson* kurtosis (Fisher=False gives ``mu4/mu2^2``);
    with ``fisher=True`` the normal distribution's value 3 is subtracted, and
    the APT salience term uses ``sqrt(clamp(kurt, min=0))``.
    """
    if weight_column.numel() == 0 or activation_column.numel() == 0:
        return torch.zeros_like(weight_column, dtype=torch.float32)

    w = weight_column.detach().reshape(-1).to(torch.float32)
    x = activation_column.detach().reshape(-1).to(torch.float32)
    if max_samples is not None and max_samples > 0 and x.numel() > max_samples:
        stride = max(1, int(math.ceil(x.numel() / max_samples)))
        x = x[::stride]
    n = x.numel()
    if n < 2:
        return torch.zeros_like(w)

    s1 = torch.zeros_like(w)
    s2 = torch.zeros_like(w)
    s3 = torch.zeros_like(w)
    s4 = torch.zeros_like(w)
    step = max(1, int(chunk))
    for start in range(0, n, step):
        xc = x[start : start + step].unsqueeze(0)  # (1, c)
        o = w.unsqueeze(1) * xc  # (d_o, c)
        o2 = o * o
        s1 += o.sum(dim=1)
        s2 += o2.sum(dim=1)
        s3 += (o2 * o).sum(dim=1)
        s4 += (o2 * o2).sum(dim=1)

    m1 = s1 / n
    m2 = s2 / n
    m3 = s3 / n
    m4 = s4 / n
    mu2 = (m2 - m1 * m1).clamp_min(eps)
    mu4 = m4 - 4.0 * m1 * m3 + 6.0 * m1 * m1 * m2 - 3.0 * m1.pow(4)
    kurt = mu4 / (mu2 * mu2)
    if fisher:
        kurt = kurt - 3.0
    return kurt


def kurtosis_term(
    kurt: torch.Tensor,
    *,
    transform: str = "sqrt",
    weight: float = 1.0,
    eps: float = 0.0,
) -> torch.Tensor:
    """``(Kurt(O))^{1/2}`` term of Equation (5)."""
    value = kurt.to(torch.float32)
    if transform == "sqrt":
        value = torch.sqrt(torch.clamp(value, min=eps))
    elif transform in ("abs", "identity", "none"):
        value = value.abs() if transform == "abs" else value
    elif transform == "log1p":
        value = torch.log1p(torch.clamp(value, min=0.0))
    elif transform == "square":
        value = value * value
    else:  # pragma: no cover - defensive
        value = torch.sqrt(torch.clamp(value, min=eps))
    return weight * value


def activation_gradient_product(
    activation: torch.Tensor,
    gradient: torch.Tensor,
    *,
    reduce: str = "sum",
    outer: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compressed activation x gradient product of Equation (5).

    ``activation`` holds ``H`` (shape ``(..., d_i)`` for the input dimension, or
    ``(..., d_o)`` for the output dimension) and ``gradient`` the corresponding
    ``dL/dH`` of the same shape.  Both are reduced over every leading dimension
    (batch and sequence) *before* the product, and the outer product between the
    output-gradient vector and the input-activation vector is formed once so it
    serves both the input-dimension and the output-dimension scores.

    Returns ``(input_dim_score, output_dim_score, product)`` where ``product``
    has shape ``(d_o, d_i)``.
    """
    h_in = _reduce_leading(activation, reduce=reduce)
    g_out = _reduce_leading(gradient, reduce=reduce)
    if not outer:
        product = (h_in * g_out).abs()
        return h_in.abs(), product, product
    product = torch.outer(g_out.abs(), h_in.abs())  # (d_o, d_i)
    return product.sum(dim=0), product.sum(dim=1), product


def _reduce_leading(tensor: torch.Tensor, reduce: str = "sum") -> torch.Tensor:
    """Sum (or mean) every dimension except the last one."""
    if tensor is None:  # pragma: no cover - defensive
        return tensor
    value = tensor
    while value.dim() > 1:
        value = value.sum(dim=0) if reduce == "sum" else value.mean(dim=0)
    return value


@torch.no_grad()
def weight_grad_scores(
    module: nn.Module,
    loss: torch.Tensor,
    *,
    names: Optional[Sequence[str]] = None,
    retain_graph: bool = True,
    include_tuning: bool = True,
    scaling: Optional[float] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Exact Equation (3) scores via autograd on the *frozen* weights.

    This is the expensive reference path: ``|W o dL/dW|`` is computed with real
    autograd on the frozen weights instead of the activation x gradient product.
    It is used for validation and for models where the frozen weights do take
    part in the graph.  Only the first occurrence of each parameter is scored
    (frozen weights are usually shared).
    """
    params: List[nn.Parameter] = []
    meta: List[Tuple[str, int, Optional[torch.Tensor]]] = []
    seen: set = set()
    for name, mod in iter_masked_linears(module, names=names):
        w = getattr(mod, "base_weight", None)
        if isinstance(w, nn.Parameter) and id(w) not in seen:
            seen.add(id(w))
            params.append(w)
            meta.append((name, HEAD, None))
        if include_tuning:
            adapter = getattr(mod, "adapter", None)
            lora_b = getattr(adapter, "lora_b", None)
            if isinstance(lora_b, nn.Parameter) and id(lora_b) not in seen:
                seen.add(id(lora_b))
                params.append(lora_b)
                meta.append((name, NEURON, None))
    if not params:
        return {}

    grads = torch.autograd.grad(
        loss, params, retain_graph=retain_graph, allow_unused=True
    )
    out: Dict[str, Dict[str, torch.Tensor]] = {}
    for (name, _kind, _unused), param, grad in zip(meta, params, grads):
        if grad is None:
            continue
        score = (param.detach().to(torch.float32) * grad.detach().to(torch.float32)).abs()
        entry = out.setdefault(name, {})
        if "base_weight" not in entry and param is getattr(
            getattr(module, "", None), "base_weight", None
        ):
            pass
        entry["weight" if param is not grad else "weight"] = score
        # rows/columns are interpreted by the caller through the shapes
        entry["row_sum"] = score.sum(dim=1)
        entry["col_sum"] = score.sum(dim=0)
    return out


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass
class SalienceConfig:
    """Hyper-parameters of the outlier-aware salience scorer."""

    beta: float = DEFAULT_BETA
    use_kurtosis: bool = True
    kurtosis_weight: float = 1.0
    kurtosis_transform: str = "sqrt"
    kurtosis_fisher: bool = True
    kurtosis_chunk: int = DEFAULT_KURTOSIS_CHUNK
    kurtosis_max_samples: Optional[int] = None
    include_frozen: bool = True
    include_tuning: bool = True
    reduce: str = "sum"
    head_grouping: bool = True
    only_pruned_kurtosis: bool = True
    dtype: torch.dtype = torch.float32
    eps: float = 1e-12

    def as_dict(self) -> Dict[str, object]:
        return dict(self.__dict__)


# ---------------------------------------------------------------------------
# exponential moving average over task batches
# ---------------------------------------------------------------------------
class MovingAverageSalience:
    """Global block scores ``S_bar`` with the 0.85/0.15 exponential moving average."""

    def __init__(self, beta: float = DEFAULT_BETA, device=None, dtype=torch.float32):
        self.beta = float(beta)
        self.device = device
        self.dtype = dtype
        self._values: Dict[str, Union[float, torch.Tensor]] = {}
        self._steps: Dict[str, int] = {}

    # -- bookkeeping -------------------------------------------------------
    def __contains__(self, name: str) -> bool:
        return name in self._values

    def __getitem__(self, name: str):
        return self._values[name]

    def get(self, name: str, default=0.0):
        return self._values.get(name, default)

    def keys(self):
        return self._values.keys()

    def items(self):
        return self._values.items()

    def __len__(self) -> int:
        return len(self._values)

    @property
    def names(self) -> List[str]:
        return list(self._values.keys())

    def as_dict(self) -> Dict[str, Union[float, torch.Tensor]]:
        return dict(self._values)

    # -- updates -----------------------------------------------------------
    def update(
        self,
        current: Dict[str, Union[float, torch.Tensor]],
        beta: Optional[float] = None,
    ) -> Dict[str, Union[float, torch.Tensor]]:
        """``S_bar <- beta * S_bar + (1 - beta) * S`` for every entry of ``current``."""
        beta = self.beta if beta is None else float(beta)
        for name, value in current.items():
            value = _to_float_tensor(value)
            previous = self._values.get(name)
            if previous is None:
                self._values[name] = value
            else:
                self._values[name] = moving_average_update(previous, value, beta)
            self._steps[name] = self._steps.get(name, 0) + 1
        return self._values

    def adjust(self, names: Iterable[str] = None) -> None:
        """Scale every stored score so that its maximum is 1 (optional helper)."""
        targets = list(self._values.keys()) if names is None else list(names)
        for name in targets:
            value = self._values.get(name)
            if value is None:
                continue
            scale = float(value.abs().max()) if torch.is_tensor(value) else abs(value)
            if scale > 0:
                self._values[name] = value / scale

    def reset(self, names: Optional[Iterable[str]] = None) -> None:
        if names is None:
            self._values.clear()
            self._steps.clear()
        else:
            for name in names:
                self._values.pop(name, None)
                self._steps.pop(name, None)

    def prune(self, keep: Iterable[str]) -> None:
        keep = set(keep)
        for name in list(self._values.keys()):
            if name not in keep:
                self._values.pop(name, None)
                self._steps.pop(name, None)

    # -- serialization -----------------------------------------------------
    def state_dict(self) -> Dict[str, object]:
        return {"beta": self.beta, "values": self.as_dict(), "steps": dict(self._steps)}

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.beta = float(state.get("beta", self.beta))
        self._values = dict(state.get("values", {}))
        self._steps = dict(state.get("steps", {}))


def _to_float_tensor(value: Union[float, torch.Tensor, int]) -> Union[float, torch.Tensor]:
    if torch.is_tensor(value):
        return value.detach().to(torch.float32)
    return float(value)


# ---------------------------------------------------------------------------
# main scorer
# ---------------------------------------------------------------------------
class OutlierAwareSalience(nn.Module):
    """Computes and tracks the outlier-aware salience of APT blocks.

    Typical use inside the training loop (Algorithm 1):

    .. code-block:: python

        loss = criterion(model(**batch), labels)
        teacher_loss = distill(model, teacher)          # optional
        total = (1 - mu) * loss + mu * teacher_loss
        total.backward()
        scores = salience.step(model)                   # cache -> EMA update
        tracker.set(scores) ...                         # optional
        blocks = selector.enumerate_blocks(salience=scores)
        manager.step(model, salience=scores, sparsity=gamma_t)
        optimizer.zero_grad()

    ``step`` only *records* scores: the optimizer update stays in the training
    loop, exactly as described by Algorithm 1.
    """

    def __init__(
        self,
        beta: float = DEFAULT_BETA,
        *,
        use_kurtosis: bool = True,
        kurtosis_weight: float = 1.0,
        kurtosis_transform: str = "sqrt",
        kurtosis_fisher: bool = True,
        kurtosis_chunk: int = DEFAULT_KURTOSIS_CHUNK,
        kurtosis_max_samples: Optional[int] = None,
        include_frozen: bool = True,
        include_tuning: bool = True,
        reduce: str = "sum",
        head_grouping: bool = True,
        only_pruned_kurtosis: bool = True,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        config: Optional[SalienceConfig] = None,
    ) -> None:
        super().__init__()
        if config is not None:
            beta = config.beta
            use_kurtosis = config.use_kurtosis
            kurtosis_weight = config.kurtosis_weight
            kurtosis_transform = config.kurtosis_transform
            kurtosis_fisher = config.kurtosis_fisher
            kurtosis_chunk = config.kurtosis_chunk
            kurtosis_max_samples = config.kurtosis_max_samples
            include_frozen = config.include_frozen
            include_tuning = config.include_tuning
            reduce = config.reduce
            head_grouping = config.head_grouping
            only_pruned_kurtosis = config.only_pruned_kurtosis
            dtype = config.dtype
        self.config = SalienceConfig(
            beta=beta,
            use_kurtosis=use_kurtosis,
            kurtosis_weight=kurtosis_weight,
            kurtosis_transform=kurtosis_transform,
            kurtosis_fisher=kurtosis_fisher,
            kurtosis_chunk=kurtosis_chunk,
            kurtosis_max_samples=kurtosis_max_samples,
            include_frozen=include_frozen,
            include_tuning=include_tuning,
            reduce=reduce,
            head_grouping=head_grouping,
            only_pruned_kurtosis=only_pruned_kurtosis,
            dtype=dtype,
        )
        self.beta = float(beta)
        self.dtype = dtype
        self._device = device
        self.moving_average = MovingAverageSalience(beta=beta, device=device, dtype=dtype)
        self.last_scores: Dict[str, Union[float, torch.Tensor]] = {}
        self._raw: Dict[str, Dict[str, torch.Tensor]] = {}

    # -- convenience -------------------------------------------------------
    @property
    def device(self) -> Optional[torch.device]:
        return self._device

    def reset(self) -> None:
        self.moving_average.reset()
        self.last_scores = {}
        self._raw = {}

    def __len__(self) -> int:
        return len(self.moving_average)

    def get(self, name: str, default=0.0):
        return self.moving_average.get(name, default)

    def as_dict(self) -> Dict[str, Union[float, torch.Tensor]]:
        return self.moving_average.as_dict()

    def values(self) -> List[Union[float, torch.Tensor]]:
        return list(self.moving_average.values())

    def set(self, scores: Dict[str, Union[float, torch.Tensor]]) -> None:
        """Overwrite the moving averages directly (used by tests/baselines)."""
        self.moving_average._values = {
            k: _to_float_tensor(v) for k, v in dict(scores).items()
        }

    # -- raw salience ------------------------------------------------------
    @torch.no_grad()
    def collect(
        self,
        model: Optional[nn.Module] = None,
        *,
        names: Optional[Sequence[str]] = None,
        shape=None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Raw (single-step) activation x gradient scores per masked linear.

        Returns a dict with the keys ``in``, ``out``, ``head``, ``neuron``,
        ``bsal`` and ``in_masked`` (kurtosis corrected input-dimension scores).
        """
        out: Dict[str, Dict[str, torch.Tensor]] = {
            "in": {},
            "out": {},
            "head": {},
            "neuron": {},
            "bsal": {},
            "in_masked": {},
        }
        if model is None:
            return out
        model_device = _module_device(model)
        device = self._device or model_device

        for name, mod in iter_masked_linears(model, names=names):
            cached_in = getattr(mod, "_cached_input", None)
            cached_out = getattr(mod, "_cached_output", None)
            cached_grad = getattr(mod, "_cached_output_grad", None)
            if cached_out is None or cached_grad is None:
                continue
            cached_out = cached_out.detach()
            cached_grad = cached_grad.detach()
            if cached_in is None:
                cached_in = cached_out

            # --- Equation (5): compressed activation x gradient product ---
            o = cached_out.to(torch.float32)
            g = cached_grad.to(torch.float32)
            if o.shape != g.shape:
                g = _match_shape(g, o)
            if self.config.reduce == "mean":
                product = torch.einsum("...i,...o->io", o.reshape(-1, o.shape[-1]).mean(0),
                                       g.reshape(-1, g.shape[-1]).mean(0)).abs()
                h_in = o.reshape(-1, o.shape[-1]).abs().mean(0)
                g_out = g.reshape(-1, g.shape[-1]).abs().mean(0)
            else:
                prod = torch.einsum("...i,...o->io", o, g).abs()
                h_in = _reduce_leading(o.abs(), reduce="sum")
                g_out = _reduce_leading(g.abs(), reduce="sum")
                product = prod if prod.dim() == 2 else torch.outer(g_out, h_in)

            out_dim_score = product.sum(dim=0).to(torch.float32)  # over blocks of the output
            in_dim_score = product.sum(dim=1).to(torch.float32)  # per input dimension
            # NOTE: ``product`` is (d_o, d_i); summing over d_o gives the input
            # dimension salience and summing over d_i the output dimension one.
            out_dim_score = product.sum(dim=1)
            in_dim_score = product.sum(dim=0)

            if not self.config.include_frozen:
                in_dim_score = torch.zeros_like(in_dim_score)
                out_dim_score = torch.zeros_like(out_dim_score)

            out["in"][name] = in_dim_score
            out["out"][name] = out_dim_score

            # --- per-head scores of this module's output groups ----------
            if self.config.head_grouping:
                heads = self._group_scores(mod, out_dim_score, kind=HEAD)
                if heads is not None:
                    out["head"][name] = heads
                neurons = self._group_scores(mod, out_dim_score, kind=NEURON)
                if neurons is not None:
                    out["neuron"][name] = neurons

            # --- adapter importance I(H_apt) = sum |W_B o dL/dW_B| -------
            if self.config.include_tuning:
                bsal = self._adapter_scores(mod)
                if bsal is not None:
                    out["bsal"][name] = bsal

            # --- kurtosis of O_{:,j} for the pruned hidden dims ----------
            if self.config.use_kurtosis:
                corrected = self._kurtosis_corrected(
                    mod, in_dim_score, cached_in, shape=shape
                )
                out["in_masked"][name] = corrected

        self._raw = out
        return out

    # -- helpers -----------------------------------------------------------
    def _group_scores(
        self, mod: nn.Module, out_dim_score: torch.Tensor, kind: int
    ) -> Optional[torch.Tensor]:
        """Reshape a per-output score into per-head / per-neuron scores."""
        group_size = int(getattr(mod, "out_group_size", 1) or 1)
        num_groups = int(getattr(mod, "num_out_groups", 0) or 0)
        if kind == HEAD:
            if group_size <= 1 or num_groups <= 1:
                return None
            if out_dim_score.numel() != group_size * num_groups:
                return None
            return out_dim_score.reshape(num_groups, group_size).sum(dim=1)
        # neurons: one output unit per group
        if group_size != 1:
            return None
        return out_dim_score

    def _adapter_scores(self, mod: nn.Module) -> Optional[torch.Tensor]:
        """``I(H_apt) = sum_{i,j} S(W_B i,j)`` from the adapter's cached gradient."""
        adapter = getattr(mod, "adapter", None)
        if adapter is None:
            return None
        lora_b = getattr(adapter, "lora_b", None)
        if lora_b is None or lora_b.grad is None:
            return None
        scaling = float(getattr(adapter, "scaling", 1.0) or 1.0)
        score = (lora_b.detach().to(torch.float32) * lora_b.grad.detach().to(torch.float32)).abs()
        row = score.sum(dim=1) * scaling  # (r_apt,) tuning-rank salience
        if not self.config.include_frozen:
            row = row
        total = float(score.sum()) * scaling
        result = torch.cat(
            [row, torch.as_tensor([total], dtype=torch.float32, device=row.device)]
        )
        return result

    def _kurtosis_corrected(
        self,
        mod: nn.Module,
        in_dim_score: torch.Tensor,
        cached_in: Optional[torch.Tensor],
        *,
        shape=None,
    ) -> torch.Tensor:
        """Add ``sqrt(Kurt(O_{j,:}))`` to the input-dimension salience."""
        if cached_in is None:
            return in_dim_score
        base_weight = getattr(mod, "base_weight", None)
        if base_weight is None:
            return in_dim_score
        columns = self._kurtosis_columns(mod, in_dim_score, shape=shape)
        if columns is None or columns.numel() == 0:
            return in_dim_score

        d_model = None
        if shape is not None:
            d_model = getattr(shape, "d_model", None)
        if d_model is not None and int(in_dim_score.numel()) != int(d_model):
            # only the hidden-dimension (input) blocks are kurtosis corrected
            return in_dim_score

        x = cached_in.detach()
        if x.dim() > 1:
            x = x.reshape(-1, x.shape[-1])
        x = x.to(torch.float32)
        weight = base_weight.detach().to(torch.float32)
        if weight.shape[1] != x.shape[-1]:
            return in_dim_score

        corrected = in_dim_score.clone().to(torch.float32)
        idx = columns.to(x.device).long()
        if idx.numel() and idx.max().item() >= x.shape[-1]:
            idx = idx[idx < x.shape[-1]]
        for j in idx.tolist():
            kurt = column_kurtosis(
                weight[:, j],
                x[:, j],
                chunk=self.config.kurtosis_chunk,
                max_samples=self.config.kurtosis_max_samples,
                fisher=self.config.kurtosis_fisher,
                eps=self.config.eps,
            )
            correction = kurtosis_term(
                kurt,
                transform=self.config.kurtosis_transform,
                weight=self.config.kurtosis_weight,
                eps=self.config.eps,
            )
            corrected[j] = corrected[j] + correction.mean()
        return corrected

    def _kurtosis_columns(
        self, mod: nn.Module, in_dim_score: torch.Tensor, *, shape=None
    ) -> Optional[torch.Tensor]:
        """Hidden-dimension indices for which extra kurtosis effort is spent.

        By default only the currently *pruned* hidden dimensions are inspected,
        because if every column of every layer were inspected the kurtosis pass
        would be ``n_layers`` times more expensive than the rest of APT's
        salience computation (see the note in this module's docstring).
        """
        mask = getattr(mod, "mask_in", None)
        if mask is None:
            return None
        values = mask.detach().reshape(-1).to(torch.float32)
        if self.config.only_pruned_kurtosis and float(values.min()) >= 1.0:
            # nothing pruned yet -> the kurtosis term cannot influence the
            # ordering of the hidden dimensions at the first pruning step.
            if shape is None:
                return None
        if self.config.only_pruned_kurtosis:
            idx = torch.nonzero(values < 0.5, as_tuple=False).reshape(-1)
            if idx.numel() == 0:
                # pruning candidates = the least salient half of the dimensions
                k = max(1, values.numel() // 2)
                order = torch.argsort(in_dim_score.to(torch.float32))[:k]
                idx = order
            return idx
        return torch.arange(values.numel(), device=values.device)

    # -- public scoring ----------------------------------------------------
    @torch.no_grad()
    def block_salience(
        self,
        model: Optional[nn.Module] = None,
        *,
        shape=None,
        blocks: Optional[Sequence[object]] = None,
        names: Optional[Sequence[str]] = None,
        raw: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    ) -> Dict[str, float]:
        """Outlier-aware salience per prunable block, keyed by block name.

        The keys match ``apt.block_selection.Block.name`` so the returned dict
        can be fed straight into the latency-saliency knapsack
        (``BlockSelector.select(...)`` / ``BlockSelector.recompute_densities``).
        """
        raw = self.collect(model, names=names, shape=shape) if raw is None else raw
        if blocks is None:
            blocks = self._enumerate_blocks(shape, raw)
        if blocks is None:
            return {}

        # aggregate the input-dimension scores of a layer into one vector
        layer_dims: Dict[int, torch.Tensor] = {}
        for name, mod in iter_masked_linears(model, names=names) if model is not None else []:
            score = raw["in"].get(name)
            if score is None:
                continue
            layer = int(getattr(mod, "layer_idx", -1))
            score = raw["in_masked"].get(name, score)
            if layer in layer_dims:
                existing = layer_dims[layer]
                if existing.numel() == score.numel():
                    layer_dims[layer] = existing + score
                else:
                    layer_dims[-1] = _add_or_keep(layer_dims.get(-1), score)
            else:
                layer_dims[layer] = score.detach().to(torch.float32)

        # neuron scores grouped per layer
        layer_neurons: Dict[int, torch.Tensor] = {}
        module_head: Dict[str, Tuple[int, str]] = {}
        module_neuron: Dict[str, int] = {}
        for name, mod in iter_masked_linears(model, names=names) if model is not None else []:
            layer = int(getattr(mod, "layer_idx", -1))
            if name in raw["head"]:
                module_head[name] = (layer, module_site(name, HEAD))
            if name in raw["neuron"]:
                module_neuron[name] = layer

        for name, scores in raw["neuron"].items():
            layer = module_neuron.get(name, -1)
            existing = layer_neurons.get(layer)
            if existing is None or existing.numel() != scores.numel():
                if existing is None:
                    layer_neurons[layer] = scores.detach().to(torch.float32)
            else:
                layer_neurons[layer] = existing + scores.detach().to(torch.float32)

        out: Dict[str, float] = {}
        for block in blocks:
            kind = int(getattr(block, "kind", 0))
            layer = int(getattr(block, "layer", -1))
            index = int(getattr(block, "index", 0))
            site = getattr(block, "site", None)
            value: Optional[torch.Tensor] = None
            if kind == HEAD:
                value = self._head_value(raw, module_head, layer, index, site)
            elif kind == NEURON:
                tensor = layer_neurons.get(layer)
                if tensor is not None and index < tensor.numel():
                    value = tensor[index]
            else:  # DIMENSION (shared across layers)
                value = self._dim_value(layer_dims, index)
            name = getattr(block, "name", None)
            if name is None:
                name = f"{kind}.{layer}.{index}"
            if value is None:
                out[str(name)] = 0.0
            else:
                out[str(name)] = float(value)
        return out

    @staticmethod
    def _head_value(raw, module_head, layer, index, site) -> Optional[torch.Tensor]:
        total = None
        for name, (mod_layer, mod_site) in module_head.items():
            if mod_layer != layer:
                continue
            if site is not None and mod_site != site:
                continue
            scores = raw["head"].get(name)
            if scores is None or index >= scores.numel():
                continue
            value = scores[index]
            total = value if total is None else total + value
        return total

    @staticmethod
    def _dim_value(layer_dims: Dict[int, torch.Tensor], index: int) -> Optional[torch.Tensor]:
        total = None
        for _layer, scores in layer_dims.items():
            if index < scores.numel():
                value = scores[index]
                total = value if total is None else total + value
        return total

    @staticmethod
    def _enumerate_blocks(shape, raw):
        if shape is None:
            return None
        try:
            from .block_selection import BlockSelector

            selector = BlockSelector(shape)
            return selector.enumerate_blocks(salience=None)
        except Exception:  # pragma: no cover - defensive
            return None

    # -- Algorithm 1 interface --------------------------------------------
    def step(
        self,
        model: Optional[nn.Module] = None,
        *,
        shape=None,
        blocks: Optional[Sequence[object]] = None,
        names: Optional[Sequence[str]] = None,
        beta: Optional[float] = None,
        add_kurtosis: Optional[bool] = None,
        update: bool = True,
    ) -> Dict[str, float]:
        """Score the model once and update the global (EMA) scores."""
        scores = self.block_salience(
            model, shape=shape, blocks=blocks, names=names
        )
        if update:
            self.moving_average.update(scores, beta=beta)
        self.last_scores = scores
        return scores

    def __call__(self, model=None, **kwargs) -> Dict[str, float]:  # type: ignore[override]
        return self.step(model, **kwargs)

    def score_model(
        self,
        model: nn.Module,
        loss: Optional[torch.Tensor] = None,
        *,
        shape=None,
        backward: bool = False,
        retain_graph: bool = False,
        names: Optional[Sequence[str]] = None,
    ) -> Dict[str, float]:
        """Convenience wrapper: optionally back-propagates then scores."""
        if backward:
            if loss is None:
                raise ValueError("a loss is required when backward=True")
            loss.backward(retain_graph=retain_graph)
        return self.step(model, shape=shape, names=names)

    # -- exact Equation (3) reference path ---------------------------------
    def exact_scores(
        self,
        model: nn.Module,
        loss: torch.Tensor,
        *,
        shape=None,
        names: Optional[Sequence[str]] = None,
        retain_graph: bool = True,
    ) -> Dict[str, float]:
        """Equation (3) scores ``|W o dL/dW|`` (no kurtosis, no compression)."""
        table = weight_grad_scores(
            model, loss, names=names, retain_graph=retain_graph
        )
        raw: Dict[str, Dict[str, torch.Tensor]] = {
            "in": {},
            "out": {},
            "head": {},
            "neuron": {},
            "bsal": {},
            "in_masked": {},
        }
        for name, entry in table.items():
            rows = entry.get("row_sum")
            cols = entry.get("col_sum")
            if rows is None or cols is None:
                continue
            raw["in"][name] = cols
            raw["out"][name] = rows
            raw["in_masked"][name] = cols
        return self.block_salience(model, shape=shape, names=names, raw=raw)

    # -- serialization -----------------------------------------------------
    def state_dict(self) -> Dict[str, object]:  # type: ignore[override]
        return {
            "beta": self.beta,
            "config": self.config.as_dict(),
            "moving_average": self.moving_average.state_dict(),
        }

    def load_state_dict(  # type: ignore[override]
        self, state: Dict[str, object], strict: bool = True
    ) -> None:
        if not state:
            return
        self.beta = float(state.get("beta", self.beta))
        if "moving_average" in state:
            self.moving_average.load_state_dict(state["moving_average"])

    # -- misc --------------------------------------------------------------
    def summary(self, top_k: int = 5) -> Dict[str, object]:
        values = self.as_dict()
        items = [(k, float(v)) for k, v in values.items()]
        items.sort(key=lambda kv: kv[1], reverse=True)
        return {
            "num_blocks": len(items),
            "top_blocks": items[:top_k],
            "bottom_blocks": items[-top_k:] if items else [],
            "beta": self.beta,
        }

    def extra_repr(self) -> str:
        return (
            f"beta={self.beta}, kurtosis={self.config.use_kurtosis}, "
            f"blocks={len(self.moving_average)}, reduce={self.config.reduce}"
        )


#: Backwards-compatible alias used by the training loop / scripts.
SalienceTracker = OutlierAwareSalience


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
def _module_device(module: nn.Module) -> torch.device:
    for param in module.parameters():
        return param.device
    for buf in module.buffers():
        return buf.device
    return torch.device("cpu")


def _match_shape(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Best-effort reshape of ``source`` so it matches ``target``'s shape."""
    if source.numel() == target.numel():
        return source.reshape(target.shape)
    if target.dim() == source.dim() and target.shape[-1] != source.shape[-1]:
        return source
    return source


def _add_or_keep(existing: Optional[torch.Tensor], value: torch.Tensor) -> torch.Tensor:
    value = value.detach().to(torch.float32)
    if existing is None:
        return value
    if existing.numel() != value.numel():
        return existing
    return existing + value


def _mean_kurtosis_correction(kurt: torch.Tensor) -> torch.Tensor:
    """``sqrt(Kurt)`` averaged over the neurons of one dimension block."""
    return torch.sqrt(torch.clamp(kurt, min=0.0)).mean()


if __name__ == "__main__":  # pragma: no cover - smoke test
    torch.manual_seed(0)

    class Dummy(nn.Module):
        def __init__(self, d_in=16, d_out=32, rank=4):
            super().__init__()
            self.base_weight = nn.Parameter(torch.randn(d_out, d_in), requires_grad=False)
            self.base_bias = nn.Parameter(torch.zeros(d_out), requires_grad=False)
            self.mask_in = torch.ones(d_in)
            self.mask_out = torch.ones(d_out)
            self.out_group_size = d_out // 4
            self.num_out_groups = 4
            self.layer_idx = 0
            adapter = nn.Module()
            adapter.scaling = 2.0
            adapter.lora_b = nn.Parameter(torch.randn(d_out, rank) * 0.1)
            adapter.lora_a = nn.Parameter(torch.randn(rank, d_in))
            adapter.rank = rank
            self.adapter = adapter
            self._cached_input = None
            self._cached_output = None
            self._cached_output_grad = None

    module = Dummy()
    x = torch.randn(8, 5, 16)
    x.requires_grad_(True)
    out = torch.nn.functional.linear(x, module.base_weight, module.base_bias)
    module._cached_input = x
    module._cached_output = out
    loss = (out**2).mean()
    loss.backward()
    module._cached_output_grad = out.grad

    scorer = OutlierAwareSalience(beta=0.85)
    raw = scorer.collect(module)
    assert raw["in"][""].shape[-1] == 16, raw["in"][""].shape
    assert raw["out"][""].shape[-1] == 32
    assert raw["head"][""].numel() == 4
    assert raw["bsal"][""].numel() == 5  # rank + total

    scores = {"head.0.1.q": 3.0, "neuron.0.4": 1.0}
    scorer.moving_average.update(scores)
    assert abs(scorer.get("head.0.1.q") - 3.0) < 1e-6
    scorer.moving_average.update(scores)
    # 0.85 * 3 + 0.15 * 3 = 3
    assert abs(scorer.get("head.0.1.q") - 3.0) < 1e-6
    scorer.moving_average.update({"head.0.1.q": 0.0})
    assert abs(scorer.get("head.0.1.q") - 0.85 * 3.0) < 1e-6

    # kurtosis of a column with heavy outliers must exceed a Gaussian column's
    gauss = torch.randn(4096)
    heavy = gauss.clone()
    heavy[0] = 50.0
    k_gauss = column_kurtosis(torch.randn(8), gauss, fisher=True)
    k_heavy = column_kurtosis(torch.randn(8), heavy, fisher=True)
    assert k_heavy.mean() > k_gauss.mean(), (k_gauss.mean(), k_heavy.mean())
    print("apt/salience.py smoke test OK")
