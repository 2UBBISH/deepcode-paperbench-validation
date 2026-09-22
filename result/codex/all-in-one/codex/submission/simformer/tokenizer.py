"""The Simformer tokenizer (Sec. 3.1 of the paper).

Every variable of the joint distribution ``x_hat = (theta, x)`` is represented by
a token that consists of

1. an *identifier* embedding that uniquely identifies the variable,
2. a *value* embedding: the scalar value is repeated to the desired
   dimensionality (Appendix, "Tokenization"),
3. optionally a *metadata* embedding (used for variables with an index, e.g. a
   time point, or for additional covariates),
4. a *condition state* embedding: a learnable vector embedding for values that
   are conditioned on (``True``) and zeros for latent values (``False``).

The four embeddings are concatenated *in that order* and linearly projected to
the token dimension of the transformer.  For function valued parameters (or
observations at arbitrary time points) the identifier is composed of a shared
(learnable) embedding and a random Fourier embedding of the element in the index
set -- this is how the Simformer remains agnostic to a particular discretization
and can be evaluated at arbitrary positions in time/space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """Fixed random Gaussian Fourier embedding of (low dimensional) indices.

    ``u -> [cos(2 pi W u), sin(2 pi W u)]`` with a frozen random matrix
    ``W ~ N(0, scale^2)``.  This is the embedding that the paper uses for
    diffusion timesteps and for the index set of function valued variables.
    """

    def __init__(self, out_dim: int, in_dim: int = 1, scale: float = 1.0,
                 seed: int = 0):
        super().__init__()
        assert out_dim % 2 == 0, "the Fourier embedding dimension must be even"
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer(
            "weight",
            torch.randn(out_dim // 2, in_dim, generator=generator) * scale,
        )
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.scale = scale

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        if u.dim() == 1:
            u = u.unsqueeze(-1)
        if u.shape[-1] != self.in_dim:
            raise ValueError(
                f"expected last dimension {self.in_dim}, got {u.shape[-1]}")
        proj = 2.0 * torch.pi * (u @ self.weight.T)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


@dataclass
class TokenizerConfig:
    n_variables: int
    d_model: int = 50
    id_dim: int = 50
    value_dim: int = 50
    cond_dim: int = 50
    metadata_dim: int = 0
    token_mlp: bool = True
    # If True the identifier embedding is a free parameter per variable.  For
    # function valued variables the identifiers have to be passed explicitly
    # (see :meth:`IdentifierEmbedding`).
    learnable_identifiers: bool = True


class Tokenizer(nn.Module):
    """Turns ``(values, condition state)`` into a sequence of tokens."""

    def __init__(self, config: TokenizerConfig):
        super().__init__()
        self.config = config
        if config.learnable_identifiers:
            self.identifier_embedding = nn.Parameter(
                0.02 * torch.randn(config.n_variables, config.id_dim))
        else:
            self.identifier_embedding = None
        # learnable vector embedding for the "conditioned" (True) state;
        # latent (False) values are projected to zeros.
        self.condition_embedding = nn.Parameter(
            0.02 * torch.randn(config.cond_dim))
        in_dim = (config.id_dim + config.value_dim + config.metadata_dim
                  + config.cond_dim)
        self.projection = nn.Linear(in_dim, config.d_model)
        if config.token_mlp:
            self.token_mlp = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, config.d_model),
            )
        else:
            self.token_mlp = None

    # ------------------------------------------------------------------ shapes
    @property
    def d_model(self) -> int:
        return self.config.d_model

    # ------------------------------------------------------------------ forward
    def forward(self,
                values: torch.Tensor,
                condition_state: torch.Tensor,
                identifiers: Optional[torch.Tensor] = None,
                metadata: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return tokens of shape ``(batch, n_variables, d_model)``.

        Parameters
        ----------
        values:
            ``(batch, n_variables)`` scalar values of every variable.  For nodes
            that are latent during training this is the *noisy* value ``x_t``,
            for conditioned nodes it is the clean value ``x_0``.
        condition_state:
            ``(batch, n_variables)`` in ``{0, 1}``.  ``1``/``True`` marks a
            variable that is conditioned on (clean), ``0``/``False`` a latent
            variable.
        identifiers:
            optional ``(batch, n_variables, id_dim)`` identifier embeddings.
            Required for function valued variables (or if the tokenizer was built
            without learnable identifiers).
        metadata:
            optional ``(batch, n_variables, metadata_dim)`` metadata embeddings.
        """
        batch, n_vars = values.shape
        cfg = self.config

        if identifiers is None:
            if self.identifier_embedding is None:
                raise ValueError(
                    "This tokenizer has no learnable identifiers, `identifiers` "
                    "has to be passed explicitly.")
            identifiers = self.identifier_embedding.unsqueeze(0).expand(
                batch, n_vars, cfg.id_dim)
        else:
            identifiers = identifiers.to(values.dtype)

        # value embedding: repeat the scalar to the requested dimensionality.
        value_embedding = (values.unsqueeze(-1)
                           * torch.ones(1, 1, cfg.value_dim,
                                        dtype=values.dtype,
                                        device=values.device))

        # condition state: learnable vector embedding for True, zeros for False.
        condition_state = condition_state.to(values.dtype).unsqueeze(-1)
        condition_embedding = condition_state * self.condition_embedding

        parts = [identifiers, value_embedding]
        if cfg.metadata_dim > 0:
            if metadata is None:
                metadata = torch.zeros(batch, n_vars, cfg.metadata_dim,
                                       dtype=values.dtype, device=values.device)
            parts.append(metadata.to(values.dtype))
        parts.append(condition_embedding)

        tokens = self.projection(torch.cat(parts, dim=-1))
        if self.token_mlp is not None:
            tokens = tokens + self.token_mlp(tokens)
        return tokens


class IdentifierEmbedding(nn.Module):
    """Identifier embeddings for function valued variables / index sets.

    An identifier is the sum of

    * a learnable embedding that is *shared* between all variables of the same
      kind (e.g. all entries of the time dependent contact rate ``beta(t)``), and
    * a random Fourier embedding of the element of the index set (e.g. the time
      point ``t``).

    A variable without an index (e.g. a global parameter) simply uses the shared
    embedding.  The distinction between "global" and "indexed" variables is made
    through ``variable_kind`` and ``include_fourier``.
    """

    def __init__(self, n_variables: int, id_dim: int, n_kinds: int,
                 use_fourier: Optional[torch.Tensor] = None,
                 index_dim: int = 1, fourier_scale: float = 1.0,
                 fourier_seed: int = 0):
        super().__init__()
        self.id_dim = id_dim
        self.shared = nn.Parameter(0.02 * torch.randn(n_kinds, id_dim))
        if use_fourier is None:
            use_fourier = torch.zeros(n_variables, dtype=torch.bool)
        self.register_buffer("use_fourier", use_fourier.bool())
        self.n_variables = n_variables
        self.index_dim = index_dim
        if bool(self.use_fourier.any()):
            if id_dim % 2 != 0:
                raise ValueError(
                    "id_dim has to be even when Fourier index embeddings are "
                    "used.")
            self.fourier = FourierFeatures(id_dim, in_dim=index_dim,
                                           scale=fourier_scale,
                                           seed=fourier_seed)
        else:
            self.fourier = None

    def forward(self, variable_kind: torch.Tensor,
                index: Optional[torch.Tensor] = None,
                use_fourier: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return ``(batch, n_variables, id_dim)`` identifier embeddings.

        ``variable_kind``: ``(n_variables,)`` long tensor with the kind of every
        variable.  ``index``: ``(batch, n_variables, index_dim)`` index values
        (only used for variables with ``use_fourier == True``).  ``use_fourier``
        can be passed explicitly, which is required when the number of variables
        of the queried problem differs from the one used during training (e.g.
        the SIRD task is evaluated on a finer grid of time points).
        """
        n_variables = int(variable_kind.shape[0])
        if use_fourier is None:
            use_fourier = self.use_fourier
        use_fourier = use_fourier.to(variable_kind.device).bool()
        if use_fourier.shape[0] != n_variables:
            raise ValueError(
                "`use_fourier` has to have one entry per variable "
                f"({use_fourier.shape[0]} != {n_variables}).")
        base = self.shared[variable_kind]                       # (n, id_dim)
        batch = 1 if index is None else index.shape[0]
        out = base.unsqueeze(0).expand(batch, -1, -1).clone()
        if self.fourier is not None and index is not None:
            fourier = self.fourier(index.reshape(-1, self.index_dim))
            fourier = fourier.reshape(batch, n_variables, self.id_dim)
            mask = use_fourier.view(1, -1, 1)
            out = torch.where(mask, out + fourier, out)
        return out
