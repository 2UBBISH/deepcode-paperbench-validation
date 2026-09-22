"""SBI tokenizer for the Simformer.

Implements the tokenizer described in Section 3.1 of "Simformer: Simulation-based
inference with probabilistic diffusion models" and the Addendum section
"Tokenization".

All variables of the joint vector ``(theta, x)`` (parameters first, then data) are
reduced to a token that is the concatenation of four embeddings, in this order:

1. an **identifier** embedding (learnable vector per variable),
2. a **value** embedding: the scalar value repeated to the desired dimensionality,
3. an optional **metadata** embedding (e.g. a random Fourier embedding of the
   index set for function-valued parameters),
4. a **condition state** embedding: the learnable per-variable embedding if the
   variable is conditioned on (condition state ``True``) and zeros otherwise.

For function-valued parameters the identifier comprises a *shared* identifier
embedding plus a random Fourier embedding of the elements of the index set
(Section 3.1).

The module is a thin wrapper on top of ``torch.nn``; the whole package uses a
PyTorch neural-network backend with NumPy for the numerical primitives in
``simformer.utils``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .utils import random_fourier_features

__all__ = [
    "TokenSpec",
    "Tokenizer",
    "FunctionValuedSpec",
    "build_benchmark_spec",
]


# --------------------------------------------------------------------------------------
# Specifications
# --------------------------------------------------------------------------------------
@dataclass
class FunctionValuedSpec:
    """Description of one function-valued parameter.

    A function-valued parameter occupies ``n_index_points`` tokens, all of which
    share the identifier embedding but receive a random Fourier embedding of
    their index value (an element of ``index_set``) as metadata.
    """

    name: str
    index_set: np.ndarray
    n_index_points: Optional[int] = None

    def __post_init__(self) -> None:
        self.index_set = np.asarray(self.index_set, dtype=np.float64).reshape(-1)
        if self.n_index_points is None:
            self.n_index_points = int(self.index_set.shape[0])
        if int(self.n_index_points) != int(self.index_set.shape[0]):
            # subsample the index set to the requested number of tokens
            idx = np.linspace(
                0, self.index_set.shape[0] - 1, int(self.n_index_points)
            ).round().astype(int)
            self.index_set = self.index_set[idx]


@dataclass
class TokenSpec:
    """Ordering and layout of the tokens for a task.

    Parameters
    ----------
    parameter_names, data_names:
        Plain scalar variable names, in the order in which they appear in the
        joint vector (``theta`` first, then ``x``).
    function_valued:
        Function-valued parameters (see :class:`FunctionValuedSpec`).
    metadata_dim:
        Dimensionality of the (optional) per-variable metadata embedding.  For
        function-valued parameters the metadata is the random Fourier embedding
        of the index point, so this is the output dimension of the RFF.
    """

    parameter_names: Sequence[str] = ()
    data_names: Sequence[str] = ()
    function_valued: Sequence[FunctionValuedSpec] = ()
    metadata_dim: int = 0
    rff_lengthscale: float = 0.2
    rff_seed: int = 0


def build_benchmark_spec(
    n_parameters: int, n_data: int
) -> TokenSpec:
    """Convenience constructor for a plain benchmark task."""
    return TokenSpec(
        parameter_names=[f"theta_{i}" for i in range(n_parameters)],
        data_names=[f"x_{i}" for i in range(n_data)],
    )


# --------------------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------------------
class Tokenizer(nn.Module):
    """Tokenizer mapping ``(theta, x)`` and condition mask ``M_C`` to tokens.

    Parameters
    ----------
    spec:
        :class:`TokenSpec` describing the task layout.
    token_dim:
        Dimensionality of a token (``50`` in the paper).
    n_parameter_variables / n_data_variables:
        Alternative (simple) way of specifying the layout when no names are used.
        The joint vector is then ``[theta_0, ..., theta_{p-1}, x_0, ..., x_{q-1}]``.
    """

    def __init__(
        self,
        spec: Optional[TokenSpec] = None,
        token_dim: int = 50,
        *,
        n_parameter_variables: Optional[int] = None,
        n_data_variables: Optional[int] = None,
    ) -> None:
        super().__init__()
        if spec is None:
            if n_parameter_variables is None or n_data_variables is None:
                raise ValueError(
                    "Provide either a TokenSpec or both "
                    "n_parameter_variables and n_data_variables."
                )
            spec = build_benchmark_spec(n_parameter_variables, n_data_variables)
        self.spec = spec
        self.token_dim = int(token_dim)

        self.scalar_var_names: List[str] = list(spec.parameter_names) + list(
            spec.data_names
        )
        self.n_scalar_vars = len(self.scalar_var_names)
        self.function_valued: List[FunctionValuedSpec] = list(spec.function_valued)
        self.n_function_tokens = int(
            sum(int(fv.n_index_points) for fv in self.function_valued)
        )
        # total number of tokens per sample
        self.n_tokens = self.n_scalar_vars + self.n_function_tokens
        # number of variables in the joint vector (function-valued parameters
        # contribute their index points as values, so they count once each)
        self.n_variables = self.n_scalar_vars + self.n_function_tokens

        self.metadata_dim = int(spec.metadata_dim)

        # summed embedding dimensionalities. The value embedding is the scalar
        # repeated to `token_dim`; the condition state embedding has `token_dim`.
        per_token_embed_dim = (
            1  # value (scalar repeated to token_dim means: token_dim values)
        )
        # For the value we use a (per-variable) affine map from the repeated
        # scalar to token_dim, which keeps the "repeat the scalar" semantics.
        self.value_dim = self.token_dim
        self.condition_dim = self.token_dim
        self.id_dim = self.token_dim
        self.metadim_used = self.metadata_dim

        # ------------------------------------------------------------------
        # 1) identifier embeddings
        # ------------------------------------------------------------------
        # scalar variables get their own identifier embedding
        self.id_embedding = nn.Embedding(self.n_scalar_vars, self.id_dim)
        # function-valued variables share one identifier embedding
        self.shared_function_id_embedding = nn.Parameter(
            torch.randn(self.id_dim) * 0.02
        )

        # ------------------------------------------------------------------
        # 3) optional metadata embedding (random Fourier embedding of index set)
        # ------------------------------------------------------------------
        if self.metadata_dim > 0 and self.n_function_tokens > 0:
            meta = np.zeros((self.n_function_tokens, self.metadata_dim), dtype=np.float32)
            offset = 0
            for fv in self.function_valued:
                feats = random_fourier_features(
                    fv.index_set,
                    out_dim=self.metadata_dim,
                    lengthscale=spec.rff_lengthscale,
                    seed=spec.rff_seed,
                )
                feats = np.asarray(feats, dtype=np.float32).reshape(
                    int(fv.n_index_points), self.metadata_dim
                )
                meta[offset : offset + int(fv.n_index_points)] = feats
                offset += int(fv.n_index_points)
            self.register_buffer("metadata", torch.as_tensor(meta))
        else:
            self.register_buffer("metadata", torch.zeros(0, 0))

        # ------------------------------------------------------------------
        # 2) value embedding: repeat the scalar to token_dim (paper) and then a
        # learnable linear projection that mixes the repeated value.
        # ------------------------------------------------------------------
        self.value_projection = nn.Linear(self.value_dim, self.token_dim)

        # ------------------------------------------------------------------
        # 4) condition-state embeddings (one learnable vector per variable)
        # ------------------------------------------------------------------
        # a single shared "conditioned" embedding vector, as in the paper the
        # learnable embedding is per variable type; we index per variable.
        self.condition_embedding = nn.Embedding(self.n_variables, self.condition_dim)

        self.reset_parameters()

    # ----------------------------------------------------------------------------------
    def reset_parameters(self) -> None:
        nn.init.normal_(self.id_embedding.weight, std=0.02)
        nn.init.normal_(self.condition_embedding.weight, std=0.02)

    # ----------------------------------------------------------------------------------
    @property
    def input_dim(self) -> int:
        """Dimensionality of the joint vector ``(theta, x)``."""
        return self.n_variables

    # ----------------------------------------------------------------------------------
    def identity_embeddings(self, batch_size: int, device=None, dtype=None) -> torch.Tensor:
        """Return the ``[B, n_tokens, id_dim]`` identifier embedding tensor."""
        id_emb = self.id_embedding.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        if self.n_function_tokens > 0:
            shared = self.shared_function_id_embedding.to(
                device=id_emb.device, dtype=id_emb.dtype
            )
            shared = shared.view(1, 1, -1).expand(
                batch_size, self.n_function_tokens, -1
            )
            id_emb = torch.cat([id_emb, shared], dim=1)
        return id_emb

    # ----------------------------------------------------------------------------------
    def forward(
        self,
        theta: torch.Tensor,
        x: torch.Tensor,
        condition_mask: Optional[torch.Tensor] = None,
        *,
        values: Optional[torch.Tensor] = None,
        function_values: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Tokenize a batch.

        Parameters
        ----------
        theta, x:
            ``[B, n_theta]`` / ``[B, n_x]`` parameter and data values.
        condition_mask:
            ``[B, n_variables]`` boolean/float tensor.  ``True`` means the
            variable is *conditioned on* (its condition-state embedding is used),
            ``False`` means the embedding is zeros (the variable is noised).
            Defaults to all-False (unconditional model).
        function_values:
            Optional dict mapping function-valued parameter name to a
            ``[B, n_index_points]`` tensor of sampled function values.

        Returns
        -------
        tokens: ``[B, n_tokens, token_dim]``
        """
        if values is not None:
            theta, x = None, values
        device = theta.device if theta is not None else x.device
        dtype = theta.dtype if theta is not None else x.dtype

        parts: List[torch.Tensor] = []
        if theta is not None and theta.shape[-1] > 0:
            parts.append(theta)
        if x is not None and x.shape[-1] > 0:
            parts.append(x)
        if function_values:
            for fv in self.function_valued:
                parts.append(function_values[fv.name])
        joint = torch.cat(parts, dim=-1)
        batch_size = joint.shape[0]
        if joint.shape[-1] != self.n_variables:
            raise ValueError(
                f"joint vector has {joint.shape[-1]} variables but tokenizer "
                f"expects {self.n_variables}"
            )

        # ---- identifier embedding ------------------------------------------------
        id_emb = self.identity_embeddings(batch_size, device=device, dtype=dtype)

        # ---- value embedding: repeat scalar to token_dim, then project ----------
        repeated = joint.unsqueeze(-1).expand(-1, -1, self.value_dim)
        val_emb = self.value_projection(repeated)

        # ---- metadata --------------------------------------------------------------
        if self.metadim_used > 0 and self.n_function_tokens > 0:
            meta = self.metadata.to(device=device, dtype=dtype).unsqueeze(0).expand(
                batch_size, -1, -1
            )
            tokens = torch.cat([id_emb, val_emb + 0.0, meta], dim=-1)
            # metadata only for the function-valued tokens: the scalar tokens
            # get zero metadata to keep the tensor rectangular.
            meta_pad = torch.zeros(
                batch_size,
                self.n_scalar_vars,
                self.metadim_used,
                device=device,
                dtype=dtype,
            )
            tokens = torch.cat([id_emb, val_emb, torch.cat([meta_pad, meta], dim=1)], dim=-1)
        else:
            tokens = torch.cat([id_emb, val_emb], dim=-1)

        # ---- condition state --------------------------------------------------------
        if condition_mask is None:
            cond = torch.zeros(
                batch_size, self.n_variables, device=device, dtype=torch.bool
            )
        else:
            cond = condition_mask.to(device=device).bool()
        idx = torch.arange(self.n_variables, device=device)
        cond_emb = self.condition_embedding(idx).unsqueeze(0).expand(
            batch_size, -1, -1
        )
        cond_emb = cond_emb * cond.unsqueeze(-1).to(cond_emb.dtype)

        tokens = torch.cat([tokens, cond_emb], dim=-1)

        # The concatenated dimension must equal token_dim: id (50) + value (50)
        # + metadata + condition (50). We map it back to token_dim with a linear
        # layer so that the token dimensionality matches the paper.
        return self.out_projection(tokens)

    # ----------------------------------------------------------------------------------
    def token_dim_total(self) -> int:
        return self.id_dim + self.value_dim + self.metadim_used + self.condition_dim
