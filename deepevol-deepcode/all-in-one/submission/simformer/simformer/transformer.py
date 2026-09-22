"""Transformer score network for the Simformer (Sec. 2.2, Sec. 3, App. A2.1).

The Simformer parameterizes the score of the joint density ``p(theta, x)`` with a
transformer that operates on the token sequence produced by
:class:`simformer.tokenizer.Tokenizer` and respects a task-specific attention mask
``M_E`` (Sec. 3.2).  The architecture follows the paper's configuration
(App. A2.1):

* token dimension ``50``,
* ``6`` transformer layers (``8`` for the Lotka-Volterra, SIRD and
  Hodgkin-Huxley tasks),
* ``4`` attention heads with attention size ``10`` (i.e. per-head key/query/value
  dimension ``10`` -> ``40`` wide attention output, projected back to ``50``),
* a widening factor of ``3`` so the feed-forward block expands to a hidden
  dimension of ``150``,
* the diffusion time is represented by a ``128``-dimensional random Gaussian
  Fourier embedding which is added -- through a linear projection -- to the
  output of every feed-forward block,
* scaled dot-product attention
  ``attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V`` (Eq. 2.1) with the mask
  ``M_E`` applied to the attention logits.

The attention mask convention mirrors :mod:`simformer.attention_masks`:
``M[i, j] = 1`` means that query token ``i`` may attend to key token ``j``
(equivalently the edge ``j -> i`` in the graphical model), and the diagonal is
always ``True`` (self attention).

The module also exposes a convenience wrapper :class:`SimformerScoreNetwork`
which couples the tokenizer with the transformer, i.e. it maps a (possibly
partially noised) joint vector ``x_hat`` and a diffusion time ``t`` directly to
the score/epsilon estimate.  This is the object used by
:mod:`simformer.training` and :mod:`simformer.sampling`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:  # pragma: no cover - torch is a hard requirement in practice
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _HAS_TORCH = False

from .utils import GaussianFourierFeatures

__all__ = [
    "DEEP_LAYER_TASKS",
    "TransformerConfig",
    "TimeFourierEmbedding",
    "MultiHeadAttention",
    "FeedForwardBlock",
    "TransformerLayer",
    "TransformerScoreNetwork",
    "ScoreNetwork",
    "SimformerScoreNetwork",
    "Simformer",
    "build_transformer",
    "build_score_network",
    "n_layers_for_task",
    "sanitize_attention_mask",
    "to_torch_mask",
    "count_parameters",
]


# --------------------------------------------------------------------------------------
# Defaults from Appendix A2.1
# --------------------------------------------------------------------------------------

DEFAULT_TOKEN_DIM = 50
DEFAULT_N_LAYERS = 6
DEFAULT_N_HEADS = 4
DEFAULT_ATTENTION_SIZE = 10  # per-head dimension -> 4 * 10 = 40 wide attention
DEFAULT_WIDENING_FACTOR = 3  # feed-forward hidden dim = 3 * 50 = 150
DEFAULT_TIME_EMBED_DIM = 128
DEFAULT_TIME_EMBED_SCALE = 16.0
DEFAULT_ACTIVATION = "gelu"

#: Tasks for which the paper increased the number of transformer layers to 8
#: ("For the Lotka-Volterra, SIR, and Hodgkin-Huxley tasks, we increased the
#: number of layers to 8.", App. A2.1).
DEEP_LAYER_TASKS = frozenset(
    {
        "lotka_volterra",
        "lotka-volterra",
        "lv",
        "sird",
        "sir",
        "hodgkin_huxley",
        "hodgkin-huxley",
        "hh",
    }
)


def n_layers_for_task(task: Optional[str], default: int = DEFAULT_N_LAYERS) -> int:
    """Return the number of transformer layers used for ``task`` (App. A2.1)."""
    if task is None:
        return default
    return 8 if str(task).lower() in DEEP_LAYER_TASKS else default


# --------------------------------------------------------------------------------------
# Mask utilities
# --------------------------------------------------------------------------------------


def to_torch_mask(
    mask: Union[np.ndarray, "torch.Tensor"], device: Any = None, dtype: Any = None
) -> "torch.Tensor":
    """Coerce ``mask`` to a boolean torch tensor (``True`` = keep the key)."""
    if not _HAS_TORCH:  # pragma: no cover
        raise ImportError("torch is required for to_torch_mask")
    if dtype is None:
        dtype = torch.bool
    if isinstance(mask, torch.Tensor):
        return mask.to(device=device, dtype=dtype)
    arr = np.asarray(mask)
    return torch.as_tensor(arr, device=device, dtype=dtype)


def sanitize_attention_mask(
    mask: Optional[Union[np.ndarray, "torch.Tensor"]],
    n_tokens: int,
    device: Any = None,
    batch_size: Optional[int] = None,
    diagonal: bool = True,
) -> Optional["torch.Tensor"]:
    """Normalize an attention mask to shape ``(B, 1, n, n)`` (bool).

    The mask may be provided as

    * ``(n, n)``           -> shared across the batch,
    * ``(B, n, n)``        -> per-sample mask,
    * ``(B, 1, n, n)``     -> already broadcastable.

    Rows that would attend to nothing are fixed by enabling the diagonal entry so
    that ``softmax`` never sees an all ``-inf`` row (the diagonal is always true
    in the paper's masks anyway).
    """
    if mask is None:
        return None
    m = to_torch_mask(mask, device=device, dtype=torch.bool)
    while m.dim() < 4:
        m = m.unsqueeze(0) if m.dim() == 2 else m.unsqueeze(1)
    if m.shape[-1] != n_tokens or m.shape[-2] != n_tokens:
        raise ValueError(
            f"attention mask has shape {tuple(m.shape)} but {n_tokens} tokens were given"
        )
    if diagonal:
        eye = torch.eye(n_tokens, device=m.device, dtype=torch.bool)
        rows_empty = ~m.any(dim=-1, keepdim=True)  # (B,1,n,1)
        m = m | (rows_empty & eye)
    if batch_size is not None and m.shape[0] == 1 and batch_size > 1:
        m = m.expand(batch_size, -1, -1, -1)
    return m.contiguous()


# --------------------------------------------------------------------------------------
# Time embedding
# --------------------------------------------------------------------------------------


def _module_base():
    return nn.Module if _HAS_TORCH else object


if _HAS_TORCH:

    class TimeFourierEmbedding(nn.Module):
        """128-dimensional random Gaussian Fourier embedding of the diffusion time.

        ``gamma(t) = [cos(2 pi t B), sin(2 pi t B)]`` with ``B ~ N(0, scale^2)``
        drawn once and kept fixed (Tancik et al., 2020 style random Fourier
        features; Sec. A2.1: "a 128-dimensional random Gaussian Fourier
        embedding").
        """

        def __init__(self, out_dim: int = DEFAULT_TIME_EMBED_DIM, scale: float = DEFAULT_TIME_EMBED_SCALE, seed: int = 0):
            super().__init__()
            self.out_dim = int(out_dim)
            self.n_freqs = (self.out_dim + 1) // 2
            rng = np.random.default_rng(seed)
            freqs = rng.normal(0.0, float(scale), size=self.n_freqs).astype(np.float32)
            self.register_buffer("freqs", torch.as_tensor(freqs))

        def forward(self, t):
            if not torch.is_tensor(t):
                t = torch.as_tensor(np.asarray(t), dtype=self.freqs.dtype, device=self.freqs.device)
            t = t.to(dtype=self.freqs.dtype, device=self.freqs.device)
            t = t.reshape(-1, 1)
            ang = 2.0 * math.pi * t * self.freqs.reshape(1, -1)
            emb = torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1)
            return emb[..., : self.out_dim]

        def extra_repr(self) -> str:  # pragma: no cover - cosmetic
            return f"out_dim={self.out_dim}, n_freqs={self.n_freqs}"


    class MultiHeadAttention(nn.Module):
        """Scaled dot-product attention with a task mask ``M_E`` (Eq. 2.1)."""

        def __init__(self, dim: int, n_heads: int = DEFAULT_N_HEADS, attention_size: int = DEFAULT_ATTENTION_SIZE, dropout: float = 0.0):
            super().__init__()
            self.dim = int(dim)
            self.n_heads = int(n_heads)
            self.head_dim = int(attention_size)
            self.inner_dim = self.n_heads * self.head_dim
            self.scale = 1.0 / math.sqrt(float(self.head_dim))
            self.qkv = nn.Linear(self.dim, 3 * self.inner_dim)
            self.out_proj = nn.Linear(self.inner_dim, self.dim)
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        def forward(self, x, mask: Optional["torch.Tensor"] = None):
            b, n, _ = x.shape
            qkv = self.qkv(x).reshape(b, n, 3, self.n_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)  # (B, n, H, D) each
            q = q.transpose(1, 2)  # (B, H, n, D)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            att = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, H, n, n)
            if mask is not None:
                att = att.masked_fill(~mask, float("-inf"))
            att = torch.softmax(att, dim=-1)
            att = self.dropout(att)
            out = torch.matmul(att, v)  # (B, H, n, D)
            out = out.transpose(1, 2).reshape(b, n, self.inner_dim)
            return self.out_proj(out)


    class FeedForwardBlock(nn.Module):
        """Feed-forward block with widening factor ``3`` (hidden size 150)."""

        def __init__(self, dim: int, widening_factor: int = DEFAULT_WIDENING_FACTOR, dropout: float = 0.0, activation: str = DEFAULT_ACTIVATION):
            super().__init__()
            hidden = int(dim) * int(widening_factor)
            self.linear_in = nn.Linear(dim, hidden)
            self.linear_out = nn.Linear(hidden, dim)
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.activation_name = activation
            self._act = self._resolve_activation(activation)

        @staticmethod
        def _resolve_activation(name: str):
            name = (name or "gelu").lower()
            if name == "gelu":
                return F.gelu
            if name in {"silu", "swish"}:
                return F.silu
            if name == "relu":
                return F.relu
            if name == "tanh":
                return torch.tanh
            raise ValueError(f"unknown activation '{name}'")

        def forward(self, x):
            return self.dropout(self.linear_out(self._act(self.linear_in(x))))


    class TransformerLayer(nn.Module):
        """Pre-norm transformer layer with diffusion-time conditioning.

        ``x <- x + attn(ln1(x), M_E)``; ``x <- x + ff(ln2(x))``; then the
        projected diffusion-time embedding is added to the feed-forward output
        (Sec. A2.1).
        """

        def __init__(
            self,
            dim: int = DEFAULT_TOKEN_DIM,
            n_heads: int = DEFAULT_N_HEADS,
            attention_size: int = DEFAULT_ATTENTION_SIZE,
            widening_factor: int = DEFAULT_WIDENING_FACTOR,
            time_embed_dim: int = DEFAULT_TIME_EMBED_DIM,
            dropout: float = 0.0,
            activation: str = DEFAULT_ACTIVATION,
            ada_layer_norm: bool = False,
        ):
            super().__init__()
            self.dim = int(dim)
            self.norm1 = nn.LayerNorm(dim)
            self.attn = MultiHeadAttention(dim, n_heads, attention_size, dropout)
            self.norm2 = nn.LayerNorm(dim)
            self.ff = FeedForwardBlock(dim, widening_factor, dropout, activation)
            self.time_proj = nn.Linear(time_embed_dim, dim)
            self.ada_layer_norm = bool(ada_layer_norm)
            if self.ada_layer_norm:
                self.ada_scale = nn.Linear(time_embed_dim, 2 * dim)
            self.reset_parameters()

        def reset_parameters(self):
            nn.init.zeros_(self.time_proj.weight)
            nn.init.zeros_(self.time_proj.bias)
            if self.ada_layer_norm:
                nn.init.zeros_(self.ada_scale.weight)
                nn.init.zeros_(self.ada_scale.bias)

        def forward(self, x, t_emb=None, mask: Optional["torch.Tensor"] = None):
            h = self.norm1(x)
            x = x + self.attn(h, mask)
            h = self.norm2(x)
            x = x + self.ff(h)
            if t_emb is not None:
                if self.ada_layer_norm:
                    scale, shift = self.ada_scale(t_emb).chunk(2, dim=-1)
                    x = x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
                x = x + self.time_proj(t_emb).unsqueeze(1)
            return x


else:  # pragma: no cover - torch missing

    class TimeFourierEmbedding:  # type: ignore
        def __init__(self, *a, **k):
            raise ImportError("torch is required for TimeFourierEmbedding")

    class MultiHeadAttention:  # type: ignore
        def __init__(self, *a, **k):
            raise ImportError("torch is required for MultiHeadAttention")

    class FeedForwardBlock:  # type: ignore
        def __init__(self, *a, **k):
            raise ImportError("torch is required for FeedForwardBlock")

    class TransformerLayer:  # type: ignore
        def __init__(self, *a, **k):
            raise ImportError("torch is required for TransformerLayer")


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------


@dataclass
class TransformerConfig:
    """Configuration of the Simformer transformer (App. A2.1)."""

    token_dim: int = DEFAULT_TOKEN_DIM
    n_layers: int = DEFAULT_N_LAYERS
    n_heads: int = DEFAULT_N_HEADS
    attention_size: int = DEFAULT_ATTENTION_SIZE
    widening_factor: int = DEFAULT_WIDENING_FACTOR
    time_embed_dim: int = DEFAULT_TIME_EMBED_DIM
    time_embed_scale: float = DEFAULT_TIME_EMBED_SCALE
    out_dim: int = 1
    dropout: float = 0.0
    activation: str = DEFAULT_ACTIVATION
    final_layer_norm: bool = True
    adaptive_layer_norm: bool = False
    time_embed_seed: int = 0
    task: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]]) -> "TransformerConfig":
        if cfg is None:
            return cls()
        if isinstance(cfg, TransformerConfig):
            return cfg
        known = {f for f in cls.__dataclass_fields__}
        payload = {k: v for k, v in dict(cfg).items() if k in known}
        return cls(**payload)

    def with_task_layers(self, task: Optional[str] = None) -> "TransformerConfig":
        """Return a copy with the paper's layer count for ``task`` (App. A2.1)."""
        task = task if task is not None else self.task
        n = n_layers_for_task(task, default=self.n_layers)
        if n == self.n_layers:
            return self
        cfg = self.to_dict()
        cfg["n_layers"] = n
        out = TransformerConfig.from_dict(cfg)
        out.task = task
        return out


# --------------------------------------------------------------------------------------
# Score network
# --------------------------------------------------------------------------------------


if _HAS_TORCH:

    class TransformerScoreNetwork(nn.Module):
        """Transformer that maps a token sequence to a per-token score estimate.

        Parameters
        ----------
        token_dim:
            Dimension of the input tokens and of the residual stream (paper: 50).
        n_layers, n_heads, attention_size, widening_factor:
            Transformer hyper-parameters (paper defaults: 6/4/10/3 -> FF hidden 150).
        time_embed_dim:
            Dimension of the random Gaussian Fourier embedding of the diffusion
            time (paper: 128).
        out_dim:
            Number of output values per token (the score of a scalar variable).
        """

        def __init__(
            self,
            token_dim: int = DEFAULT_TOKEN_DIM,
            n_layers: int = DEFAULT_N_LAYERS,
            n_heads: int = DEFAULT_N_HEADS,
            attention_size: int = DEFAULT_ATTENTION_SIZE,
            widening_factor: int = DEFAULT_WIDENING_FACTOR,
            time_embed_dim: int = DEFAULT_TIME_EMBED_DIM,
            out_dim: int = 1,
            dropout: float = 0.0,
            activation: str = DEFAULT_ACTIVATION,
            final_layer_norm: bool = True,
            adaptive_layer_norm: bool = False,
            time_embed_scale: float = DEFAULT_TIME_EMBED_SCALE,
            time_embed_seed: int = 0,
            input_projection: bool = False,
        ):
            super().__init__()
            self.token_dim = int(token_dim)
            self.n_layers = int(n_layers)
            self.n_heads = int(n_heads)
            self.attention_size = int(attention_size)
            self.widening_factor = int(widening_factor)
            self.time_embed_dim = int(time_embed_dim)
            self.out_dim = int(out_dim)

            self.time_embed = TimeFourierEmbedding(time_embed_dim, time_embed_scale, time_embed_seed)
            self.layers = nn.ModuleList(
                [
                    TransformerLayer(
                        dim=self.token_dim,
                        n_heads=self.n_heads,
                        attention_size=self.attention_size,
                        widening_factor=self.widening_factor,
                        time_embed_dim=self.time_embed_dim,
                        dropout=dropout,
                        activation=activation,
                        ada_layer_norm=adaptive_layer_norm,
                    )
                    for _ in range(self.n_layers)
                ]
            )
            self.final_norm = nn.LayerNorm(self.token_dim) if final_layer_norm else nn.Identity()
            self.out_proj = nn.Linear(self.token_dim, self.out_dim)
            self.reset_parameters()

        # -- initialization ---------------------------------------------------------
        def reset_parameters(self) -> None:
            """GPT-style init: normal(0, 0.02) for linear weights, zeros for biases."""
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
            for layer in self.layers:
                layer.reset_parameters()

        # -- forward ----------------------------------------------------------------
        def forward(self, tokens, t, attention_mask=None) -> "torch.Tensor":
            """Compute the score estimate for every token.

            Parameters
            ----------
            tokens: ``(B, n, token_dim)``
            t: scalar, ``(B,)`` or ``(B, 1)`` diffusion time (same convention as the SDE)
            attention_mask: ``(n, n)``, ``(B, n, n)`` or broadcastable boolean mask
                where ``True`` means "query may attend to key".

            Returns
            -------
            ``(B, n, out_dim)`` score (epsilon) estimates.
            """
            if not torch.is_tensor(tokens):
                tokens = torch.as_tensor(np.asarray(tokens), dtype=torch.float32)
            b, n, _ = tokens.shape
            if not torch.is_tensor(t):
                t = torch.as_tensor(np.asarray(t), dtype=tokens.dtype, device=tokens.device)
            t = t.to(device=tokens.device, dtype=tokens.dtype).reshape(-1)
            if t.shape[0] == 1 and b > 1:
                t = t.expand(b)

            t_emb = self.time_embed(t)  # (B, time_embed_dim)
            mask = sanitize_attention_mask(attention_mask, n, device=tokens.device, batch_size=b)

            x = tokens
            for layer in self.layers:
                x = layer(x, t_emb, mask)
            x = self.final_norm(x)
            return self.out_proj(x)

        def score(self, tokens, t, attention_mask=None) -> "torch.Tensor":
            """Alias of :meth:`forward` returning ``(B, n)`` scores."""
            out = self.forward(tokens, t, attention_mask=attention_mask)
            return out[..., 0] if out.shape[-1] == 1 else out

        @property
        def ff_hidden_dim(self) -> int:
            return self.token_dim * self.widening_factor

        def extra_repr(self) -> str:  # pragma: no cover - cosmetic
            return (
                f"token_dim={self.token_dim}, n_layers={self.n_layers}, n_heads={self.n_heads}, "
                f"attention_size={self.attention_size}, ff_hidden={self.ff_hidden_dim}, "
                f"time_embed_dim={self.time_embed_dim}"
            )


else:  # pragma: no cover

    class TransformerScoreNetwork:  # type: ignore
        def __init__(self, *a, **k):
            raise ImportError("torch is required for TransformerScoreNetwork")


#: Aliases matching the nomenclature used in the paper / other modules.
ScoreNetwork = TransformerScoreNetwork
ScoreNet = TransformerScoreNetwork


def build_transformer(
    task: Optional[str] = None,
    token_dim: int = DEFAULT_TOKEN_DIM,
    n_layers: Optional[int] = None,
    n_heads: int = DEFAULT_N_HEADS,
    attention_size: int = DEFAULT_ATTENTION_SIZE,
    widening_factor: int = DEFAULT_WIDENING_FACTOR,
    time_embed_dim: int = DEFAULT_TIME_EMBED_DIM,
    out_dim: int = 1,
    **kwargs,
) -> "TransformerScoreNetwork":
    """Build a :class:`TransformerScoreNetwork` with the paper's configuration.

    If ``n_layers`` is ``None`` the paper's task-dependent layer count is used
    (8 layers for Lotka-Volterra / SIR(D) / Hodgkin-Huxley, else 6).
    """
    if n_layers is None:
        n_layers = n_layers_for_task(task, default=DEFAULT_N_LAYERS)
    return TransformerScoreNetwork(
        token_dim=token_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        attention_size=attention_size,
        widening_factor=widening_factor,
        time_embed_dim=time_embed_dim,
        out_dim=out_dim,
        **kwargs,
    )


# --------------------------------------------------------------------------------------
# Tokenizer + transformer wrapper
# --------------------------------------------------------------------------------------


if _HAS_TORCH:

    class SimformerScoreNetwork(nn.Module):
        """Tokenizer + transformer: joint vector ``x_hat`` and time ``t`` -> score.

        This is the end-to-end score model ``s_phi^{M_E}(x_hat_t^{M_C}, t)`` of
        Sec. 3.3.  The tokenizer embeds the (partially noised) values together
        with their condition states; the transformer mixes tokens according to
        ``M_E``.

        The joint vector is ordered ``[theta_1..theta_p, x_1..x_n, f_1..f_k]``
        where the last block contains the (scalar) values of function-valued
        parameter tokens.
        """

        def __init__(self, tokenizer, transformer: Optional["TransformerScoreNetwork"] = None, **transformer_kwargs):
            super().__init__()
            self.tokenizer = tokenizer
            if transformer is None:
                transformer = build_transformer(token_dim=getattr(tokenizer, "token_dim", DEFAULT_TOKEN_DIM), **transformer_kwargs)
            self.transformer = transformer
            self.n_parameters = int(getattr(tokenizer, "n_parameter_variables", 0) or 0)
            self.n_data = int(getattr(tokenizer, "n_data_variables", 0) or 0)
            spec = getattr(tokenizer, "spec", None)
            if spec is not None and self.n_parameters == 0:
                self.n_parameters = len(getattr(spec, "parameter_names", ()) or ())
                self.n_data = len(getattr(spec, "data_names", ()) or ())
            self.n_function_tokens = int(getattr(tokenizer, "n_function_tokens", 0) or 0)

        # -- helpers ---------------------------------------------------------------
        @property
        def input_dim(self) -> int:
            return int(getattr(self.tokenizer, "input_dim", self.n_parameters + self.n_data))

        @property
        def token_dim(self) -> int:
            return int(getattr(self.transformer, "token_dim", DEFAULT_TOKEN_DIM))

        def split_joint(self, x_joint):
            """Split ``x_joint`` into ``(theta, x, function_values)`` blocks."""
            theta = x_joint[..., : self.n_parameters]
            x = x_joint[..., self.n_parameters : self.n_parameters + self.n_data]
            func = x_joint[..., self.n_parameters + self.n_data :]
            return theta, x, func

        # -- forward ---------------------------------------------------------------
        def forward(
            self,
            theta_or_joint,
            t,
            condition_mask=None,
            attention_mask=None,
            x=None,
            function_values=None,
        ):
            """Score estimate for the joint vector.

            ``theta_or_joint`` is either the full joint vector (``x`` is then
            ``None``) or the parameter block (``x`` must be provided).
            """
            if x is None:
                theta, x, func = self.split_joint(theta_or_joint)
                if function_values is None:
                    function_values = func if func.shape[-1] > 0 else None
            else:
                theta = theta_or_joint
            tokens = self.tokenizer(
                theta,
                x,
                condition_mask=condition_mask,
                function_values=function_values,
            )
            out = self.transformer(tokens, t, attention_mask=attention_mask)
            if out.shape[-1] == 1:
                return out[..., 0]
            return out

        #: alias used by training/sampling code
        score = forward

        def tokens(self, x_joint, condition_mask=None, function_values=None):
            """Convenience: tokenize a joint vector."""
            theta, x, func = self.split_joint(x_joint)
            if function_values is None:
                function_values = func if func.shape[-1] > 0 else None
            return self.tokenizer(theta, x, condition_mask=condition_mask, function_values=function_values)

        def make_score_fn(
            self,
            condition_mask=None,
            attention_mask=None,
            function_values=None,
            keep_conditioned: bool = True,
        ) -> Callable[["torch.Tensor", Any], "torch.Tensor"]:
            """Return ``score_fn(x_hat, t)`` for the reverse SDE sampler.

            When ``keep_conditioned`` is ``True`` the returned function first
            overwrites the conditioned entries of ``x_hat`` with
            ``conditioned_values`` (if given) so that the sampler can simply
            integrate the whole vector with the observed variables clamped by the
            caller (Sec. 3.3).
            """

            def score_fn(x_hat, t):
                return self.forward(
                    x_hat,
                    t,
                    condition_mask=condition_mask,
                    attention_mask=attention_mask,
                    function_values=function_values,
                )

            return score_fn

        def extra_repr(self) -> str:  # pragma: no cover - cosmetic
            return f"n_parameters={self.n_parameters}, n_data={self.n_data}, n_tokens={getattr(self.tokenizer, 'n_tokens', '?')}"


else:  # pragma: no cover

    class SimformerScoreNetwork:  # type: ignore
        def __init__(self, *a, **k):
            raise ImportError("torch is required for SimformerScoreNetwork")


#: ``Simformer`` alias for the end-to-end model.
Simformer = SimformerScoreNetwork


def build_score_network(
    task: Optional[str] = None,
    spec=None,
    token_dim: int = DEFAULT_TOKEN_DIM,
    n_layers: Optional[int] = None,
    **kwargs,
) -> "SimformerScoreNetwork":
    """Build the end-to-end Simformer score network for a task/``TokenSpec``.

    ``spec`` is a :class:`simformer.tokenizer.TokenSpec`.  ``task`` only selects
    the paper's layer count (App. A2.1).
    """
    from .tokenizer import Tokenizer

    tokenizer = Tokenizer(spec=spec, token_dim=token_dim) if spec is not None else Tokenizer(token_dim=token_dim)
    return SimformerScoreNetwork(
        tokenizer,
        task=task,
        n_layers=n_layers,
        token_dim=token_dim,
        **kwargs,
    )


def count_parameters(model) -> int:
    """Number of trainable parameters of a torch module."""
    if not _HAS_TORCH:  # pragma: no cover
        return 0
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
