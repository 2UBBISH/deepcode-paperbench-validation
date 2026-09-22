"""Embedding networks that condense high-dimensional data into a single token.

Simformer (``Sec. 3.1`` and ``Appendix Sec. A3.2``) represents every scalar variable of the
joint vector ``(theta, x)`` as one token.  When observations are high dimensional this becomes
computationally demanding, so -- exactly as in the gravitational-wave benchmark of
Hermans et al. (2022) -- a *specialized embedding network* may be used to compress a whole
measurement vector into a single token::

    "specialized embedding networks, commonly used in SBI algorithms and trained end-to-end
     (Lueckmann et al., 2017; Chan et al., 2018; Radev et al., 2020), can be efficiently
     integrated here by condensing complex data into a single token (e.g. we demonstrate this
     on a gravitational waves example in Appendix Sec. A3.2). This reduces computational
     complexity but loses direct control over dependencies and condition states for individual
     data elements."                                                            -- Sec. 3.1

Gravitational-wave setup (Appendix Sec. A3.2): ``theta in R^2`` (the two black-hole masses) and
two high-dimensional measurements ``x_1, x_2 in R^8192`` from two different detectors.  Because
learning the likelihood is much harder than the posterior in this regime, only the conditionals
``p(theta | x1, x2)``, ``p(theta | x1)`` and ``p(theta | x2)`` are targeted.  This file provides

* :class:`CNNEmbedding` -- a convolutional embedding network (1D convs + pooling) mapping a
  vector of length ``L`` (e.g. 8192) to a low-dimensional embedding,
* :class:`MLPEmbedding` and :class:`DeepSetEmbedding` -- alternatives,
* :func:`build_embedding_net` -- factory,
* :class:`EmbeddingTokenizer` -- turns ``(theta, x_1, ..., x_K)`` into a short token sequence
  (one token per parameter plus one token per *active* data block), keeping the Simformer token
  layout ``[identifier | value | metadata | condition state]``,
* :func:`targeted_condition_masks` / :func:`targeted_targets` -- helpers restricting the
  condition-mask distribution to the targeted (partial) posteriors,
* :func:`build_embedding_score_network` / :func:`build_gravitational_waves_model` -- glue to the
  transformer score network of :mod:`simformer.transformer`.

The module is written so that PyTorch is a *soft* dependency (as elsewhere in this code base):
importing it without torch works, instantiating the modules raises a clear ``ImportError``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:  # pragma: no cover - exercised implicitly when torch is installed
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch missing
    torch = None  # type: ignore
    nn = None  # type: ignore
    _HAS_TORCH = False


if _HAS_TORCH:

    class _Module(nn.Module):  # type: ignore[misc]
        """Base module alias (``torch.nn.Module`` when torch is available)."""

else:  # pragma: no cover - torch missing

    class _Module:
        """Placeholder base class used when torch is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "simformer.embedding_nets requires PyTorch. Install torch to use embedding networks."
            )


__all__ = [
    "DEFAULT_TOKEN_DIM",
    "DEFAULT_CNN_CHANNELS",
    "DEFAULT_KERNEL_SIZE",
    "DEFAULT_EMBED_DIM",
    "GRAVITATIONAL_WAVES_N_PARAMETERS",
    "GRAVITATIONAL_WAVES_MEASUREMENT_DIM",
    "GRAVITATIONAL_WAVES_N_MEASUREMENTS",
    "GRAVITATIONAL_WAVES_TARGETS",
    "CNNEmbedding",
    "MLPEmbedding",
    "DeepSetEmbedding",
    "build_embedding_net",
    "EmbeddingTokenizer",
    "EmbeddingSpec",
    "targeted_targets",
    "targeted_condition_masks",
    "build_embedding_score_network",
    "build_gravitational_waves_model",
    "compress_measurements",
]

ArrayLike = Union[np.ndarray, Any]

# --------------------------------------------------------------------------------------
# defaults (Sec. A2.1 architecture: token dim 50, 6 transformer layers)
# --------------------------------------------------------------------------------------
DEFAULT_TOKEN_DIM = 50
DEFAULT_CNN_CHANNELS: Tuple[int, ...] = (8, 16, 32, 64)
DEFAULT_KERNEL_SIZE = 5
DEFAULT_EMBED_DIM = 16
DEFAULT_MAX_POOLINGS = 8
DEFAULT_MIN_LENGTH_AFTER_POOL = 8.0

# Gravitational waves benchmark (Hermans et al. 2022; Sec. A3.2)
GRAVITATIONAL_WAVES_N_PARAMETERS = 2
GRAVITATIONAL_WAVES_MEASUREMENT_DIM = 8192
GRAVITATIONAL_WAVES_N_MEASUREMENTS = 2
GRAVITATIONAL_WAVES_TARGETS: Tuple[str, ...] = ("x1_x2", "x1", "x2")


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise ImportError(
            "PyTorch is required for simformer.embedding_nets (embedding networks / tokenizer)."
        )


def _n_pooling_stages(length: int, min_length: float = DEFAULT_MIN_LENGTH_AFTER_POOL,
                      max_poolings: int = DEFAULT_MAX_POOLINGS) -> int:
    """Number of halving (pooling) stages that still keeps ``>= min_length`` samples."""
    stages, current = 0, float(length)
    while current > min_length and stages < max_poolings:
        current /= 2.0
        stages += 1
    return stages


def _as_2d(x: ArrayLike) -> Any:
    """Ensure a measurement array has a leading batch dimension."""
    if isinstance(x, np.ndarray) and x.ndim == 1:
        return x[None, :]
    if _HAS_TORCH and torch.is_tensor(x) and x.dim() == 1:
        return x.unsqueeze(0)
    return x


# ======================================================================================
# Embedding networks
# ======================================================================================
if _HAS_TORCH:

    class _ConvBlock1d(nn.Module):
        """Conv1d -> normalization -> activation -> average pooling."""

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = DEFAULT_KERNEL_SIZE,
            pool: bool = True,
            activation: str = "gelu",
            norm: str = "group",
        ) -> None:
            super().__init__()
            padding = kernel_size // 2
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
            self.activation = nn.GELU() if activation in ("gelu", "GELU") else nn.SiLU()
            groups = max(1, min(8, out_channels // 4 if out_channels >= 4 else 1))
            if norm == "group":
                self.norm = nn.GroupNorm(groups, out_channels)
            elif norm == "layer":
                self.norm = nn.GroupNorm(1, out_channels)
            else:
                self.norm = nn.Identity()
            self.pool = nn.AvgPool1d(2) if pool else nn.Identity()

        def forward(self, x: Any) -> Any:
            x = self.conv(x)
            x = self.norm(x)
            x = self.activation(x)
            return self.pool(x)

    class CNNEmbedding(nn.Module):
        """1D convolutional embedding network condensing ``x in R^L`` into one vector.

        Follows the standard SBI practice of using a CNN embedding net for high dimensional
        observations (Lueckmann et al. 2017; Hermans et al. 2022): a stack of conv / norm /
        activation / average-pooling blocks followed by global average pooling and a linear
        head.  The network is fully convolutional plus a global pooling, so it accepts any
        input length ``>= 1`` (the number of pooling stages is derived from ``input_dim``).

        Parameters
        ----------
        input_dim:
            Length ``L`` of a single measurement vector (``8192`` for gravitational waves).
        out_dim:
            Dimensionality of the produced embedding (``16`` in the paper's GW setup; it is
            projected to the transformer token dimension when tokenized).
        channels:
            Conv channel widths.
        kernel_size:
            Kernel width of all convolutions.
        n_channels:
            Number of physical channels of the input (1 for a real time series).
        """

        def __init__(
            self,
            input_dim: int,
            out_dim: int = DEFAULT_EMBED_DIM,
            channels: Sequence[int] = DEFAULT_CNN_CHANNELS,
            kernel_size: int = DEFAULT_KERNEL_SIZE,
            n_channels: int = 1,
            activation: str = "gelu",
            norm: str = "group",
            input_normalization: str = "none",
            dropout: float = 0.0,
            n_blocks: Optional[int] = None,
        ) -> None:
            super().__init__()
            self.input_dim = int(input_dim)
            self.out_dim = int(out_dim)
            self.n_channels = int(n_channels)
            self.kernel_size = int(kernel_size)
            self.input_normalization = str(input_normalization)

            if n_blocks is None:
                n_blocks = max(1, _n_pooling_stages(self.input_dim))
            self.n_blocks = int(n_blocks)

            channels = list(channels) if len(channels) else list(DEFAULT_CNN_CHANNELS)
            while len(channels) < self.n_blocks:
                channels.append(channels[-1] * 2)
            channels = channels[: self.n_blocks]

            blocks: List[Any] = []
            c_in = self.n_channels
            for i, c_out in enumerate(channels):
                blocks.append(
                    _ConvBlock1d(
                        c_in,
                        int(c_out),
                        kernel_size=kernel_size,
                        pool=(i < self.n_blocks - 1) or True,
                        activation=activation,
                        norm=norm,
                    )
                )
                c_in = int(c_out)
            self.blocks = nn.ModuleList(blocks)

            self.head = nn.Sequential(
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(c_in, self.out_dim),
            )
            self.output_norm = nn.LayerNorm(self.out_dim)

            if self.input_normalization == "global":
                self.register_buffer("input_mean", torch.zeros(n_channels, 1))
                self.register_buffer("input_std", torch.ones(n_channels, 1))

            self.reset_parameters()

        # ------------------------------------------------------------------
        def reset_parameters(self) -> None:
            for module in self.modules():
                if isinstance(module, nn.Conv1d):
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        def set_normalization(self, mean: ArrayLike, std: ArrayLike) -> None:
            """Store per-channel standardization statistics (used when ``"global"``)."""
            _require_torch()
            mean_t = torch.as_tensor(np.asarray(mean, dtype=np.float32)).reshape(self.n_channels, -1)
            std_t = torch.as_tensor(np.asarray(std, dtype=np.float32)).reshape(self.n_channels, -1)
            std_t = torch.clamp(std_t, min=1e-6)
            if self.input_normalization == "global":
                self.input_mean.copy_(mean_t)
                self.input_std.copy_(std_t)

        def _reshape_input(self, x: Any) -> Any:
            x = _as_2d(x)
            x = torch.as_tensor(x)
            x = x.to(dtype=next(self.parameters()).dtype)
            if x.dim() == 2:  # (B, L) -> (B, 1, L)
                x = x.unsqueeze(1)
            elif x.dim() == 3 and x.shape[-1] == self.input_dim and x.shape[1] != self.n_channels:
                x = x.transpose(1, 2)  # (B, L, C) -> (B, C, L)
            return x

        def forward(self, x: Any) -> Any:
            x = self._reshape_input(x)
            if self.input_normalization == "global" and hasattr(self, "input_mean"):
                x = (x - self.input_mean) / self.input_std
            elif self.input_normalization == "per_sample":
                mean = x.mean(dim=-1, keepdim=True)
                std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
                x = (x - mean) / std
            for block in self.blocks:
                x = block(x)
            x = x.mean(dim=-1)  # global average pooling -> (B, C)
            x = self.head(x)
            return self.output_norm(x)

    class MLPEmbedding(nn.Module):
        """Flatten-and-MLP embedding (simple alternative to the CNN)."""

        def __init__(
            self,
            input_dim: int,
            out_dim: int = DEFAULT_EMBED_DIM,
            hidden_dims: Sequence[int] = (128, 64),
            activation: str = "gelu",
            dropout: float = 0.0,
            input_normalization: str = "none",
        ) -> None:
            super().__init__()
            self.input_dim = int(input_dim)
            self.out_dim = int(out_dim)
            self.input_normalization = str(input_normalization)
            act = nn.GELU() if activation in ("gelu", "GELU") else nn.SiLU()
            layers: List[Any] = []
            c_in = self.input_dim
            for h in hidden_dims:
                layers += [nn.Linear(c_in, int(h)), act]
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                c_in = int(h)
            layers.append(nn.Linear(c_in, self.out_dim))
            self.net = nn.Sequential(*layers)
            self.output_norm = nn.LayerNorm(self.out_dim)
            self.register_buffer("input_mean", torch.zeros(self.input_dim))
            self.register_buffer("input_std", torch.ones(self.input_dim))

        def forward(self, x: Any) -> Any:
            x = torch.as_tensor(_as_2d(x)).to(dtype=next(self.parameters()).dtype)
            x = x.reshape(x.shape[0], -1)
            if self.input_normalization == "global":
                x = (x - self.input_mean) / self.input_std
            return self.output_norm(self.net(x))

    class DeepSetEmbedding(nn.Module):
        """Permutation-invariant embedding: per-element MLP + mean pooling (deep sets)."""

        def __init__(
            self,
            input_dim: int,
            out_dim: int = DEFAULT_EMBED_DIM,
            element_dim: int = 1,
            hidden_dims: Sequence[int] = (64, 64),
            activation: str = "gelu",
            one_hot_index: bool = False,
        ) -> None:
            super().__init__()
            self.input_dim = int(input_dim)
            self.out_dim = int(out_dim)
            self.element_dim = int(element_dim)
            self.one_hot_index = bool(one_hot_index)
            act = nn.GELU() if activation in ("gelu", "GELU") else nn.SiLU()
            in_dim = self.element_dim + (1 if self.one_hot_index else 0)
            layers: List[Any] = []
            c_in = in_dim
            for h in hidden_dims:
                layers += [nn.Linear(c_in, int(h)), act]
                c_in = int(h)
            layers.append(nn.Linear(c_in, self.out_dim))
            self.phi = nn.Sequential(*layers)
            self.rho = nn.Sequential(nn.Linear(self.out_dim, self.out_dim), nn.LayerNorm(self.out_dim))
            idx = torch.arange(int(input_dim), dtype=torch.float32).unsqueeze(1)
            self.register_buffer("index", idx / max(1.0, float(input_dim) - 1.0))

        def forward(self, x: Any) -> Any:
            x = torch.as_tensor(_as_2d(x)).to(dtype=next(self.parameters()).dtype)
            x = x.reshape(x.shape[0], self.input_dim, self.element_dim)
            if self.one_hot_index:
                idx = self.index.unsqueeze(0).expand(x.shape[0], -1, -1)
                x = torch.cat([x, idx], dim=-1)
            h = self.phi(x).mean(dim=1)
            return self.rho(h)

else:  # pragma: no cover - torch missing: stubs with clear errors

    class CNNEmbedding(_Module):  # type: ignore[no-redef]
        """Stub raising ``ImportError`` when PyTorch is unavailable."""

    class MLPEmbedding(_Module):  # type: ignore[no-redef]
        """Stub raising ``ImportError`` when PyTorch is unavailable."""

    class DeepSetEmbedding(_Module):  # type: ignore[no-redef]
        """Stub raising ``ImportError`` when PyTorch is unavailable."""


def build_embedding_net(
    kind: str = "cnn",
    input_dim: int = GRAVITATIONAL_WAVES_MEASUREMENT_DIM,
    out_dim: int = DEFAULT_EMBED_DIM,
    **kwargs: Any,
) -> Any:
    """Factory returning an embedding network of the requested ``kind``.

    Parameters
    ----------
    kind: ``"cnn"`` (default), ``"mlp"`` or ``"deepset"``.
    input_dim: length of a single measurement vector.
    out_dim: embedding dimensionality (before projection to the token dimension).
    kwargs: forwarded to the chosen embedding class.
    """
    _require_torch()
    kind = str(kind).lower()
    if kind in ("cnn", "conv", "resnet", "conv1d"):
        return CNNEmbedding(input_dim=input_dim, out_dim=out_dim, **kwargs)
    if kind in ("mlp", "flatten"):
        return MLPEmbedding(input_dim=input_dim, out_dim=out_dim, **kwargs)
    if kind in ("deepset", "deepsets", "set"):
        return DeepSetEmbedding(input_dim=input_dim, out_dim=out_dim, **kwargs)
    raise ValueError(f"Unknown embedding network kind: {kind!r}")


def compress_measurements(net: Any, x: ArrayLike, batch_size: Optional[int] = None) -> np.ndarray:
    """Apply an embedding network to NumPy measurements and return a NumPy embedding."""
    _require_torch()
    x = _as_2d(np.asarray(x, dtype=np.float32))
    outputs: List[np.ndarray] = []
    bs = batch_size or x.shape[0] or 1
    with torch.no_grad():
        for start in range(0, max(1, x.shape[0]), bs):
            chunk = torch.as_tensor(x[start : start + bs], dtype=torch.float32)
            outputs.append(net(chunk).cpu().numpy())
    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0,), dtype=np.float32)


# ======================================================================================
# Embedding tokenizer
# ======================================================================================
@dataclass
class EmbeddingSpec:
    """Description of an embedding-tokenizer layout.

    Parameters
    ----------
    parameter_names:
        Names of the scalar parameters (``theta``) -- one token each.
    data_dims:
        Length of each *data block*; every block is compressed into a single token by its
        embedding network (``(8192, 8192)`` for the two gravitational-wave detectors).
    data_names:
        Optional names of the data blocks (defaults to ``x1``, ``x2``, ...).
    embed_dim:
        Output dimensionality of a single embedding network before projection.
    token_dim:
        Simformer token dimensionality (50, Sec. A2.1).
    embedding_kind:
        ``"cnn"``, ``"mlp"`` or ``"deepset"`` (used when no network is supplied).
    share_embedding:
        If ``True`` all blocks share one embedding network (the paper's two detectors are
        measurements of the same physical type); otherwise one network per block.
    metadata_dim:
        Size of the (optional) metadata slot of a token; ``0`` disables it.
    """

    parameter_names: Sequence[str] = ()
    data_dims: Sequence[int] = ()
    data_names: Sequence[str] = ()
    embed_dim: int = DEFAULT_EMBED_DIM
    token_dim: int = DEFAULT_TOKEN_DIM
    embedding_kind: str = "cnn"
    share_embedding: bool = True
    metadata_dim: int = 0
    embedding_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.parameter_names = tuple(self.parameter_names)
        self.data_dims = tuple(int(d) for d in self.data_dims)
        if not self.data_names:
            self.data_names = tuple(f"x{i + 1}" for i in range(len(self.data_dims)))
        else:
            self.data_names = tuple(self.data_names)
            if len(self.data_names) != len(self.data_dims):
                raise ValueError("data_names and data_dims must have the same length")

    @property
    def n_parameters(self) -> int:
        return len(self.parameter_names)

    @property
    def n_data_blocks(self) -> int:
        return len(self.data_dims)

    @property
    def n_tokens(self) -> int:
        return self.n_parameters + self.n_data_blocks

    @property
    def input_dim(self) -> int:
        """Total size of the flattenend joint vector ``(theta, x_1, ..., x_K)``."""
        return self.n_parameters + int(sum(self.data_dims))


class EmbeddingTokenizer(_Module):
    """Tokenizer condensing each high-dimensional data block into a single token.

    Token layout follows Sec. 3.1: every token concatenates an *identifier* embedding, a
    *value* representation, an (optional) *metadata* slot and the *condition state*, and is then
    projected to ``token_dim``.  The first ``n_parameters`` tokens correspond to the scalar
    parameters; the remaining tokens correspond to the *active* data blocks (one token per
    block, produced by the embedding network).

    Parameters
    ----------
    spec:
        :class:`EmbeddingSpec` describing the layout (an int is interpreted as the number of
        scalar parameters with ``data_dims=(8192, 8192)``).
    embedding_nets:
        Optional sequence of embedding networks (or a single shared one).
    token_dim:
        Overrides ``spec.token_dim``.
    """

    def __init__(
        self,
        spec: Union[EmbeddingSpec, int, None] = None,
        embedding_nets: Optional[Any] = None,
        token_dim: Optional[int] = None,
        *,
        data_dims: Optional[Sequence[int]] = None,
        n_parameters: Optional[int] = None,
    ) -> None:
        _require_torch()
        super().__init__()
        if isinstance(spec, int):
            spec = EmbeddingSpec(
                parameter_names=tuple(f"theta{i + 1}" for i in range(int(spec))),
                data_dims=tuple(data_dims) if data_dims is not None
                else (GRAVITATIONAL_WAVES_MEASUREMENT_DIM,) * GRAVITATIONAL_WAVES_N_MEASUREMENTS,
            )
        if spec is None:
            n_parameters = n_parameters if n_parameters is not None else GRAVITATIONAL_WAVES_N_PARAMETERS
            spec = EmbeddingSpec(
                parameter_names=tuple(f"theta{i + 1}" for i in range(int(n_parameters))),
                data_dims=tuple(data_dims) if data_dims is not None else (),
            )
        if token_dim is not None:
            spec.token_dim = int(token_dim)
        self.spec = spec

        dim = int(spec.token_dim)
        self._token_dim = dim
        self.metadim_used = int(spec.metadata_dim)

        # --- identifiers ---------------------------------------------------------------
        self.parameter_id_embedding = nn.Embedding(max(1, spec.n_parameters), dim)
        self.data_id_embedding = nn.Embedding(max(1, spec.n_data_blocks), dim)

        # --- value representations -----------------------------------------------------
        self.parameter_value_projection = nn.Linear(1, dim)
        self.data_value_projection = nn.Linear(int(spec.embed_dim), dim)

        # --- condition state -----------------------------------------------------------
        n_positions = max(1, spec.n_tokens)
        self.condition_embedding = nn.Embedding(n_positions, dim)

        # --- embedding networks --------------------------------------------------------
        nets: List[Any] = []
        if embedding_nets is None:
            if spec.share_embedding and spec.n_data_blocks > 0:
                nets.append(
                    build_embedding_net(
                        kind=spec.embedding_kind,
                        input_dim=int(spec.data_dims[0]),
                        out_dim=int(spec.embed_dim),
                        **spec.embedding_kwargs,
                    )
                )
            else:
                for length in spec.data_dims:
                    nets.append(
                        build_embedding_net(
                            kind=spec.embedding_kind,
                            input_dim=int(length),
                            out_dim=int(spec.embed_dim),
                            **spec.embedding_kwargs,
                        )
                    )
        elif isinstance(embedding_nets, (list, tuple)):
            nets = list(embedding_nets)
        else:
            nets = [embedding_nets]
        self.embedding_nets = nn.ModuleList(nets)
        self.share_embedding = bool(spec.share_embedding and len(nets) == 1)

        # --- final projection to token_dim (identifier|value|metadata|condition) -------
        n_slots = 3 + (1 if self.metadim_used > 0 else 0)
        self.out_projection = nn.Linear(n_slots * dim, dim)

        if self.metadim_used > 0:
            self.register_buffer(
                "metadata_buffer", torch.zeros(max(1, spec.n_tokens), self.metadim_used)
            )
        self.reset_parameters()

    # ----------------------------------------------------------------------------------
    # introspection
    # ----------------------------------------------------------------------------------
    @property
    def token_dim(self) -> int:
        return self._token_dim

    @property
    def n_parameters(self) -> int:
        return self.spec.n_parameters

    @property
    def n_parameter_variables(self) -> int:
        """Number of parameter (theta) variables -- used to split a joint vector."""
        return self.spec.n_parameters

    @property
    def n_data_variables(self) -> int:
        """Number of raw data dimensions (sum over blocks)."""
        return int(sum(self.spec.data_dims))

    @property
    def n_data_tokens(self) -> int:
        return self.spec.n_data_blocks

    @property
    def n_function_tokens(self) -> int:
        return 0

    @property
    def n_variables(self) -> int:
        return self.spec.n_tokens

    @property
    def n_tokens(self) -> int:
        return self.spec.n_tokens

    @property
    def input_dim(self) -> int:
        return self.spec.input_dim

    @property
    def value_dim(self) -> int:
        return self.spec.input_dim

    @property
    def condition_dim(self) -> int:
        return int(self._token_dim)

    @property
    def id_dim(self) -> int:
        return int(self._token_dim)

    # ----------------------------------------------------------------------------------
    # helpers
    # ----------------------------------------------------------------------------------
    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def split_data(self, x: ArrayLike) -> List[Any]:
        """Split ``x`` into a list of per-block arrays of shape ``(B, dim_i)``."""
        dims = list(self.spec.data_dims)
        if isinstance(x, (list, tuple)):
            blocks = []
            for i, block in enumerate(x):
                block = _as_2d(np.asarray(block, dtype=np.float32)) if not (
                    _HAS_TORCH and torch.is_tensor(block)
                ) else _as_2d(block)
                if len(dims) and i < len(dims) and block.shape[-1] != dims[i]:
                    raise ValueError(
                        f"Block {i} has last dimension {block.shape[-1]}, expected {dims[i]}"
                    )
                blocks.append(block)
            return blocks
        arr = x if (_HAS_TORCH and torch.is_tensor(x)) else np.asarray(x, dtype=np.float32)
        arr = _as_2d(arr)
        if arr.shape[-1] != sum(dims):
            raise ValueError(
                f"Flattened data has last dimension {arr.shape[-1]}, expected {sum(dims)}"
            )
        blocks, start = [], 0
        for length in dims:
            blocks.append(arr[..., start : start + length])
            start += length
        return blocks

    def _resolve_active(self, active_blocks: Optional[Any], batch_size: int) -> Any:
        """Return a long tensor of active block indices ``(n_active,)``."""
        n = self.spec.n_data_blocks
        if active_blocks is None:
            idx = list(range(n))
        elif isinstance(active_blocks, (int, np.integer)):
            idx = [int(active_blocks)]
        elif isinstance(active_blocks, str):
            names = list(self.spec.data_names)
            if active_blocks not in names:
                raise ValueError(f"Unknown data block name {active_blocks!r}; have {names}")
            idx = [names.index(active_blocks)]
        else:
            arr = np.asarray(active_blocks)
            if arr.dtype == bool:
                if arr.ndim > 1:
                    # per-sample masks: require an identical active set across the batch
                    uniq = np.unique(arr.astype(np.int8), axis=0)
                    if uniq.shape[0] > 1:
                        raise ValueError(
                            "EmbeddingTokenizer requires the same active data blocks for the whole "
                            "batch; call it once per conditional type instead."
                        )
                    arr = uniq[0].astype(bool)
                idx = list(np.nonzero(arr)[0])
            else:
                idx = [int(v) for v in arr.reshape(-1)]
        for i in idx:
            if not 0 <= i < n:
                raise ValueError(f"Active block index {i} out of range for {n} blocks")
        return torch.as_tensor(idx, dtype=torch.long)

    def _embed_theta(self, theta: Any, condition: Any) -> Any:
        """Tokenize the scalar parameters -> ``(B, n_parameters, token_dim)``."""
        theta = torch.as_tensor(theta).to(dtype=next(self.parameters()).dtype)
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        if self.spec.n_parameters == 0:
            return theta.new_zeros((theta.shape[0], 0, self.token_dim))
        if theta.shape[-1] != self.spec.n_parameters:
            raise ValueError(
                f"theta has last dimension {theta.shape[-1]}, expected {self.spec.n_parameters}"
            )
        B = theta.shape[0]
        ids = torch.arange(self.spec.n_parameters, device=theta.device)
        id_emb = self.parameter_id_embedding(ids).unsqueeze(0).expand(B, -1, -1)
        val = self.parameter_value_projection(theta.unsqueeze(-1))
        cond = self.condition_embedding(ids).unsqueeze(0).expand(B, -1, -1)
        cond = cond * condition.unsqueeze(-1).to(cond.dtype)
        parts = [id_emb, val]
        if self.metadim_used > 0:
            parts.append(torch.zeros_like(id_emb)[..., : self.metadim_used])
        parts.append(cond)
        return self.out_projection(torch.cat(parts, dim=-1))

    def _embed_block(self, block: Any, net_index: int, block_index: int, condition: Any) -> Any:
        """Compress one data block into one token -> ``(B, 1, token_dim)``."""
        block = torch.as_tensor(block).to(dtype=next(self.parameters()).dtype)
        if block.dim() == 1:
            block = block.unsqueeze(0)
        B = block.shape[0]
        device = block.device
        if self.share_embedding:
            net = self.embedding_nets[0]
        elif net_index < len(self.embedding_nets):
            net = self.embedding_nets[net_index]
        else:
            net = self.embedding_nets[net_index % max(1, len(self.embedding_nets))]
        emb = net(block)
        value = self.data_value_projection(emb).unsqueeze(1)
        id_idx = torch.as_tensor([block_index], device=device, dtype=torch.long)
        id_emb = self.data_id_embedding(id_idx).unsqueeze(0).expand(B, -1, -1)
        cond_idx = torch.as_tensor(
            [self.spec.n_parameters + block_index], device=device, dtype=torch.long
        )
        cond = self.condition_embedding(cond_idx).unsqueeze(0).expand(B, -1, -1)
        cond = cond * condition.unsqueeze(-1).to(cond.dtype)
        parts = [id_emb, value]
        if self.metadim_used > 0:
            parts.append(torch.zeros_like(id_emb)[..., : self.metadim_used])
        parts.append(cond)
        return self.out_projection(torch.cat(parts, dim=-1))

    # ----------------------------------------------------------------------------------
    # main entry points
    # ----------------------------------------------------------------------------------
    def forward(
        self,
        theta: ArrayLike,
        x: Optional[ArrayLike] = None,
        condition_mask: Optional[Any] = None,
        active_blocks: Optional[Any] = None,
        *,
        values: Optional[ArrayLike] = None,
        function_values: Optional[Any] = None,
        return_condition_mask: bool = False,
    ) -> Any:
        """Build the token sequence.

        Parameters
        ----------
        theta: ``(B, n_parameters)`` parameter values (or the full joint vector if ``x`` is None).
        x: data measurements: ``(B, sum(data_dims))`` or a sequence of ``(B, dim_i)`` blocks.
        condition_mask: ``(B, n_tokens)`` (or ``(n_tokens,)``) condition states; ``1`` marks a
            *conditioned* (observed) token, ``0`` a latent one.  Defaults to all latent, i.e. the
            unconditional joint model.
        active_blocks: subset of data blocks to include (see :meth:`_resolve_active`).
        """
        if x is None and theta is not None:
            joint = torch.as_tensor(_as_2d(theta))
            theta = joint[..., : self.spec.n_parameters]
            x = joint[..., self.spec.n_parameters :]
        theta = torch.as_tensor(_as_2d(theta))
        B = theta.shape[0]
        blocks = self.split_data(x) if len(self.spec.data_dims) else []
        active = self._resolve_active(active_blocks, B)
        active_list = [int(i) for i in active.tolist()]

        n_tokens = self.spec.n_parameters + len(active_list)
        if condition_mask is None:
            cond = torch.zeros(B, n_tokens, device=theta.device)
        else:
            cond = torch.as_tensor(condition_mask).to(theta.device)
            if cond.dtype != torch.bool:
                cond = cond > 0.5
            cond = cond.to(torch.float32)
            if cond.dim() == 1:
                cond = cond.unsqueeze(0).expand(B, -1)
            if cond.shape[1] != n_tokens:
                raise ValueError(
                    f"condition_mask has {cond.shape[1]} entries, expected {n_tokens}"
                )

        theta_cond = cond[:, : self.spec.n_parameters]
        tokens = [self._embed_theta(theta, theta_cond)]
        for j, block_index in enumerate(active_list):
            block_cond = cond[:, self.spec.n_parameters + j]
            tokens.append(
                self._embed_block(blocks[block_index], j, block_index, block_cond)
            )
        out = torch.cat(tokens, dim=1) if len(tokens) > 1 else tokens[0]
        if return_condition_mask:
            return out, cond
        return out

    __call__ = forward

    def tokens_of_theta(self, theta: ArrayLike, condition_mask: Optional[Any] = None) -> Any:
        """Only the parameter tokens (used for targeted conditionals / diagnostics)."""
        theta = torch.as_tensor(_as_2d(theta))
        B = theta.shape[0]
        if condition_mask is None:
            cond = torch.zeros(B, self.spec.n_parameters, device=theta.device)
        else:
            cond = torch.as_tensor(condition_mask).to(device=theta.device, dtype=torch.float32)
        return self._embed_theta(theta, cond)

    def parameter_condition_masks(self, batch_size: int = 1, value: float = 0.0) -> np.ndarray:
        """Condition mask (NumPy) marking all parameter tokens with ``value``."""
        mask = np.zeros((batch_size, self.spec.n_parameters), dtype=np.float32)
        mask[:] = value
        return mask

    def identity_embeddings(self, batch_size: int = 1, device: Optional[Any] = None,
                            dtype: Optional[Any] = None) -> Any:
        """Stacked identifier embeddings of all tokens (debugging utility)."""
        _require_torch()
        device = device or next(self.parameters()).device
        ids = [
            self.parameter_id_embedding(torch.arange(self.spec.n_parameters, device=device))
        ] if self.spec.n_parameters else []
        if self.spec.n_data_blocks:
            ids.append(self.data_id_embedding(torch.arange(self.spec.n_data_blocks, device=device)))
        out = torch.cat(ids, dim=0)
        if dtype is not None:
            out = out.to(dtype)
        return out.unsqueeze(0).expand(batch_size, -1, -1)

    def token_dim_total(self) -> int:
        """Dimensionality of the concatenated (pre-projection) token representation."""
        n_slots = 3 + (1 if self.metadim_used > 0 else 0)
        return n_slots * self.token_dim

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"n_parameters={self.spec.n_parameters}, data_dims={self.spec.data_dims}, "
            f"token_dim={self.token_dim}"
        )


# ======================================================================================
# Targeted conditionals (Sec. A3.2)
# ======================================================================================
def targeted_targets(
    names: Sequence[str] = GRAVITATIONAL_WAVES_TARGETS,
    data_names: Optional[Sequence[str]] = None,
    n_blocks: int = GRAVITATIONAL_WAVES_N_MEASUREMENTS,
) -> Dict[str, List[int]]:
    """Map target names (``"x1_x2"``, ``"x1"``, ``"x2"``) to active data-block indices.

    Only the conditionals ``p(theta | x1, x2)``, ``p(theta | x1)`` and ``p(theta | x2)`` are
    targeted in the gravitational-wave experiment (Sec. A3.2): the condition-mask distribution
    is restricted to those masks.
    """
    data_names = list(data_names) if data_names is not None else [f"x{i + 1}" for i in range(n_blocks)]
    out: Dict[str, List[int]] = {}
    for name in names:
        parts = [p for p in str(name).replace(",", "_").split("_") if p]
        idx: List[int] = []
        for p in parts:
            if p in data_names:
                idx.append(data_names.index(p))
            elif p.isdigit():
                idx.append(int(p) - 1)
            else:
                raise ValueError(f"Cannot resolve data block {p!r} in target {name!r}")
        out[str(name)] = idx
    return out


def targeted_condition_masks(
    tokenizer: EmbeddingTokenizer,
    targets: Optional[Union[Sequence[str], Dict[str, Sequence[int]]]] = None,
    batch_size: int = 1,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Build the condition masks / active-block pairs of the targeted posteriors.

    Returns a dict ``name -> {"condition_mask": (B, n_tokens) float32, "active_blocks": (k,)}``
    where the parameter tokens stay latent (``0``) and the active data-block tokens are
    conditioned (``1``).
    """
    if targets is None:
        targets = targeted_targets(data_names=tokenizer.spec.data_names)
    elif not isinstance(targets, dict):
        targets = targeted_targets(names=list(targets), data_names=tokenizer.spec.data_names)
    out: Dict[str, Dict[str, np.ndarray]] = {}
    n_params = tokenizer.spec.n_parameters
    for name, idx in targets.items():
        idx = list(int(i) for i in idx)
        n_tokens = n_params + len(idx)
        mask = np.zeros((batch_size, n_tokens), dtype=np.float32)
        mask[:, n_params:] = 1.0
        out[str(name)] = {
            "condition_mask": mask,
            "active_blocks": np.asarray(idx, dtype=np.int64),
        }
    return out


# ======================================================================================
# Model glue
# ======================================================================================
def build_embedding_score_network(
    tokenizer: EmbeddingTokenizer,
    n_layers: int = 6,
    n_heads: int = 4,
    attention_size: int = 10,
    widening_factor: int = 3,
    time_embed_dim: int = 128,
    out_dim: int = 1,
    attention_mask: Optional[Any] = None,
    **kwargs: Any,
) -> Any:
    """Wrap an :class:`EmbeddingTokenizer` in the Simformer transformer score network.

    The embedding network is trained end-to-end with the score network (Sec. 3.1 / A3.2).
    """
    _require_torch()
    from .transformer import TransformerScoreNetwork, SimformerScoreNetwork

    transformer = TransformerScoreNetwork(
        token_dim=tokenizer.token_dim,
        n_layers=int(n_layers),
        n_heads=int(n_heads),
        attention_size=int(attention_size),
        widening_factor=int(widening_factor),
        time_embed_dim=int(time_embed_dim),
        out_dim=int(out_dim),
        **kwargs,
    )
    model = SimformerScoreNetwork(tokenizer, transformer)
    model.attention_mask = attention_mask
    return model


def build_gravitational_waves_model(
    n_parameters: int = GRAVITATIONAL_WAVES_N_PARAMETERS,
    measurement_dim: int = GRAVITATIONAL_WAVES_MEASUREMENT_DIM,
    n_detectors: int = GRAVITATIONAL_WAVES_N_MEASUREMENTS,
    embed_dim: int = DEFAULT_EMBED_DIM,
    token_dim: int = DEFAULT_TOKEN_DIM,
    embedding_kind: str = "cnn",
    share_embedding: bool = True,
    n_layers: int = 6,
    n_heads: int = 4,
    attention_size: int = 10,
    widening_factor: int = 3,
    time_embed_dim: int = 128,
    attention_mask: Optional[Any] = None,
    **kwargs: Any,
) -> Any:
    """Simformer with a convolutional embedding net for the gravitational-wave task.

    ``theta in R^2`` (black-hole masses) and ``n_detectors`` measurements of size
    ``measurement_dim`` (``2 x 8192`` in Sec. A3.2) are each compressed into a single token.
    """
    spec = EmbeddingSpec(
        parameter_names=tuple(f"theta{i + 1}" for i in range(int(n_parameters))),
        data_dims=tuple([int(measurement_dim)] * int(n_detectors)),
        embed_dim=int(embed_dim),
        token_dim=int(token_dim),
        embedding_kind=str(embedding_kind),
        share_embedding=bool(share_embedding),
    )
    tokenizer = EmbeddingTokenizer(spec)
    return build_embedding_score_network(
        tokenizer,
        n_layers=int(n_layers),
        n_heads=int(n_heads),
        attention_size=int(attention_size),
        widening_factor=int(widening_factor),
        time_embed_dim=int(time_embed_dim),
        attention_mask=attention_mask,
        **kwargs,
    )
