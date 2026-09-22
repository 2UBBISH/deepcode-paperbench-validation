"""Neural network building blocks for SAPG actors and critics.

The paper (Appendix B.1-B.3) specifies the following architectures:

* **AllegroKuka tasks** (Appendix B.1): "We use a Gaussian policy where the mean
  network is an LSTM with 1 layer containing 768 hidden units. The observation is
  also passed through an MLP of with hidden layer dimensions 768 x 512 x 256 and an
  ELU activation before being input to the LSTM. The sigma for the Gaussian is a
  fixed learnable vector independent of input observation."
* **Shadow Hand** (Appendix B.2): "the mean network is an MLP with hidden layers
  dimensions 512 x 512 x 256 x 128 and an ELU activation."
* **Allegro Hand** (Appendix B.3): "the mean network is an MLP with hidden layers
  dimensions 512 x 256 x 128 and an ELU activation."

Appendix B.3 Note: "In case of experiments with entropy based exploration, each
block of environments has it's own learnable vector sigma which enable policies for
different blocks to have different entropies."

This module deliberately contains *only* the parameterised building blocks.  The
per-policy conditioning on ``phi_j`` is implemented in :mod:`sapg.models.actor` /
:mod:`sapg.models.critic`, where the same shared backbone ``B_theta`` / ``C_psi``
is used by every policy and the latent ``phi_j`` is injected.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

__all__ = [
    "ACTIVATIONS",
    "build_activation",
    "init_weights",
    "MLP",
    "LSTM",
    "RecurrentBackbone",
    "LearnableSigma",
    "make_backbone",
    "mlp_units_for_task",
    "log_std_to_sigma",
    "sigma_to_log_std",
]


# --------------------------------------------------------------------------------------
# Activations
# --------------------------------------------------------------------------------------
ACTIVATIONS = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "leaky_relu": nn.LeakyReLU,
    "identity": nn.Identity,
}


def build_activation(name: str = "elu") -> nn.Module:
    """Return a *fresh* activation module for ``name`` (default ELU, Appendix B)."""
    key = str(name).lower()
    if key not in ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{name}'. Available: {sorted(ACTIVATIONS)}"
        )
    return ACTIVATIONS[key]()


# --------------------------------------------------------------------------------------
# Weight initialisation
# --------------------------------------------------------------------------------------
def init_weights(module: nn.Module, gain: float = 1.0) -> nn.Module:
    """Orthogonal init for linear/recurrent weights, zeros for biases.

    The paper does not specify an initialiser; orthogonal initialisation is the
    standard choice for the ``rl_games``-style recurrent PPO trainers that the
    AllegroKuka / DexPBT codebase (which this paper builds upon) uses.
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LSTM, nn.GRU)):
        for name, param in module.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                nn.init.orthogonal_(param, gain=gain)
            elif "bias" in name:
                nn.init.zeros_(param)
    return module


# --------------------------------------------------------------------------------------
# Feed-forward backbone
# --------------------------------------------------------------------------------------
class MLP(nn.Module):
    """Multi-layer perceptron with ELU activations (Appendix B.1-B.3).

    Parameters
    ----------
    input_dim:
        Dimensionality of the (conditioned) input.
    units:
        Hidden layer widths, e.g. ``(768, 512, 256)`` for AllegroKuka,
        ``(512, 512, 256, 128)`` for Shadow Hand, ``(512, 256, 128)`` for Allegro Hand.
    output_dim:
        Optional output projection width.  When ``None`` the module returns the last
        hidden activation (used as a shared trunk feeding the policy head).
    activation:
        Activation name (paper uses ELU).
    use_layer_norm:
        Optional LayerNorm after each hidden layer (off by default; not used by the
        paper or the reference codebase).
    """

    def __init__(
        self,
        input_dim: int,
        units: Sequence[int] = (768, 512, 256),
        output_dim: Optional[int] = None,
        activation: str = "elu",
        use_layer_norm: bool = False,
        gain: float = math.sqrt(2.0),
    ) -> None:
        super().__init__()
        units = tuple(int(u) for u in units)
        dims: List[int] = [int(input_dim), *units]
        if output_dim is not None:
            dims.append(int(output_dim))

        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            is_last = i == len(dims) - 2
            if not is_last or output_dim is None:
                layers.append(build_activation(activation))
                if use_layer_norm:
                    layers.append(nn.LayerNorm(dims[i + 1]))

        self.input_dim = int(input_dim)
        self.output_dim = dims[-1]
        self.net = nn.Sequential(*layers)
        self.apply(lambda m: init_weights(m, gain=gain))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def out_features(self) -> int:
        return self.output_dim


# --------------------------------------------------------------------------------------
# Recurrent backbone
# --------------------------------------------------------------------------------------
class LSTM(nn.Module):
    """Wrapper around :class:`torch.nn.LSTM` with an rl_games-like API.

    Appendix B.1: "an LSTM with 1 layer containing 768 hidden units".
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 768,
        num_layers: int = 1,
        output_dim: Optional[int] = None,
        dropout: float = 0.0,
        gain: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.lstm = nn.LSTM(
            input_size=int(input_dim),
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=dropout if int(num_layers) > 1 else 0.0,
        )
        self.head = (
            nn.Linear(self.hidden_size, int(output_dim)) if output_dim is not None else None
        )
        self.apply(lambda m: init_weights(m, gain=gain))

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run the LSTM (batch-first) and return ``(output, (h, c))``.

        ``masks`` zeroes the carried hidden state of environments that finished an
        episode on the previous step (auto-reset masking convention).
        """
        if hidden_state is None:
            hidden_state = self.init_hidden(x.shape[0], x.device, x.dtype)
        if masks is not None:
            hidden_state = self.mask_hidden(hidden_state, masks)
        out, new_hidden = self.lstm(x, hidden_state)
        if self.head is not None:
            out = self.head(out)
        return out, new_hidden

    # ------------------------------------------------------------------ helpers
    def init_hidden(
        self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shape = (self.num_layers, int(batch_size), self.hidden_size)
        return (
            torch.zeros(shape, device=device, dtype=dtype),
            torch.zeros(shape, device=device, dtype=dtype),
        )

    @staticmethod
    def mask_hidden(
        hidden_state: Tuple[torch.Tensor, torch.Tensor], masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h, c = hidden_state
        if masks.dim() == 1:
            masks = masks.view(1, -1, 1)
        elif masks.dim() == 2:
            masks = masks.unsqueeze(0)
        return h * masks, c * masks

    @classmethod
    def from_config(
        cls,
        input_dim: int,
        layer_spec: Iterable,
        output_dim: Optional[int] = None,
        **kwargs,
    ) -> "LSTM":
        """Build from an ``rl_games``-style ``[[hidden, 'elu'], ...]`` spec."""
        units: List[int] = []
        for entry in layer_spec:
            if isinstance(entry, (list, tuple)):
                units.append(int(entry[0]))
            else:
                units.append(int(entry))
        if not units:
            raise ValueError("Empty LSTM layer spec")
        return cls(
            input_dim=input_dim,
            hidden_size=units[0],
            num_layers=max(1, len(units)),
            output_dim=output_dim,
            **kwargs,
        )


class RecurrentBackbone(nn.Module):
    """MLP trunk + LSTM head with a uniform ``(x, hidden_state, masks)`` signature."""

    def __init__(self, trunk: nn.Module, recurrent: LSTM) -> None:
        super().__init__()
        self.trunk = trunk
        self.recurrent = recurrent
        self.hidden_size = recurrent.hidden_size
        self.num_layers = recurrent.num_layers
        self.is_recurrent = True

    @property
    def out_features(self) -> int:
        if self.recurrent.head is not None:
            return self.recurrent.head.out_features
        return self.hidden_size

    def forward(self, x: torch.Tensor, hidden_state=None, masks=None):
        feats = self.trunk(x)
        squeeze = False
        if feats.dim() == 2:
            feats = feats.unsqueeze(1)
            squeeze = True
        out, new_hidden = self.recurrent(feats, hidden_state=hidden_state, masks=masks)
        if squeeze:
            out = out.squeeze(1)
        return out, new_hidden


def make_backbone(
    input_dim: int,
    mlp_units: Sequence[int] = (768, 512, 256),
    activation: str = "elu",
    use_lstm: bool = False,
    lstm_hidden_size: int = 768,
    lstm_num_layers: int = 1,
    output_dim: Optional[int] = None,
) -> nn.Module:
    """Compose the shared backbone ``B_theta`` / ``C_psi`` specified in Appendix B.

    * ``use_lstm=False`` -> :class:`MLP` over ``mlp_units``.
    * ``use_lstm=True``  -> :class:`MLP` over ``mlp_units`` followed by an
      :class:`LSTM` (Appendix B.1: MLP 768x512x256 -> LSTM with 768 hidden units).
    """
    if not bool(use_lstm):
        return MLP(
            input_dim=input_dim,
            units=mlp_units,
            output_dim=output_dim,
            activation=activation,
        )

    units = tuple(int(u) for u in mlp_units)
    trunk_out = units[-1] if units else int(input_dim)
    trunk = MLP(
        input_dim=input_dim,
        units=mlp_units,
        output_dim=trunk_out,
        activation=activation,
    )
    recurrent = LSTM(
        input_dim=trunk_out,
        hidden_size=lstm_hidden_size,
        num_layers=lstm_num_layers,
        output_dim=output_dim,
    )
    return RecurrentBackbone(trunk, recurrent)


# --------------------------------------------------------------------------------------
# Learnable, input-independent Gaussian sigma (Appendix B.1 + B.3 note)
# --------------------------------------------------------------------------------------
class LearnableSigma(nn.Module):
    """Fixed learnable vector sigma independent of the input observation.

    Appendix B.1: "The sigma for the Gaussian is a fixed learnable vector
    independent of input observation."

    Appendix B.3 note: "In case of experiments with entropy based exploration, each
    block of environments has it's own learnable vector sigma".  Passing
    ``num_policies > 1`` allocates one such vector per policy (block); the returned
    sigma is selected with ``policy_index``.

    The parameter is stored in log-space (``log_std``) for numerical stability.
    """

    def __init__(
        self,
        action_dim: int,
        init_sigma: float = 1.0,
        num_policies: int = 1,
        learnable: bool = True,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.num_policies = max(1, int(num_policies))
        self.learnable = bool(learnable)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        init_log_std = sigma_to_log_std(torch.full((self.action_dim,), float(init_sigma)))
        shape = (
            (self.num_policies, self.action_dim)
            if self.num_policies > 1
            else (self.action_dim,)
        )
        values = init_log_std.expand(shape).clone().contiguous()
        if self.learnable:
            self.log_std = nn.Parameter(values)
        else:
            self.register_buffer("log_std", values)

    # ------------------------------------------------------------------ helpers
    def log_std_for(self, policy_index: Optional[int] = None) -> torch.Tensor:
        if self.log_std.dim() == 1:
            return self.log_std
        idx = int(policy_index) if policy_index is not None else 0
        return self.log_std[idx % self.log_std.shape[0]]

    def sigma_for(self, policy_index: Optional[int] = None) -> torch.Tensor:
        return log_std_to_sigma(self.log_std_for(policy_index))

    @property
    def sigma(self) -> torch.Tensor:
        return self.sigma_for(0)

    def forward(
        self, batch_size: int, policy_index: Optional[int] = None
    ) -> torch.Tensor:
        return self.sigma_for(policy_index).unsqueeze(0).expand(int(batch_size), -1)

    # ------------------------------------------------------------------- entropy
    def entropy(self, policy_index: Optional[int] = None) -> torch.Tensor:
        """Differential entropy of a diagonal Gaussian (sum over action dims)."""
        log_std = self.log_std_for(policy_index).clamp(self.log_std_min, self.log_std_max)
        return (
            float(self.action_dim) * 0.5 * (1.0 + math.log(2.0 * math.pi))
            + log_std.sum()
        )

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"action_dim={self.action_dim}, num_policies={self.num_policies}"


# --------------------------------------------------------------------------------------
# sigma <-> log_std conversions
# --------------------------------------------------------------------------------------
def sigma_to_log_std(sigma: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.clamp(sigma, min=1e-8))


def log_std_to_sigma(log_std: torch.Tensor) -> torch.Tensor:
    return torch.exp(log_std)


# --------------------------------------------------------------------------------------
# Task -> architecture lookup (Appendix B.1-B.3)
# --------------------------------------------------------------------------------------
_MLP_UNITS_BY_TASK = {
    "allegro_kuka": (768, 512, 256),
    "regrasping": (768, 512, 256),
    "regrasp": (768, 512, 256),
    "throw": (768, 512, 256),
    "reorientation": (768, 512, 256),
    "shadow_hand": (512, 512, 256, 128),
    "shadowhand": (512, 512, 256, 128),
    "allegro_hand": (512, 256, 128),
    "allegrohand": (512, 256, 128),
}


def mlp_units_for_task(task: str) -> Tuple[int, ...]:
    """Return the Appendix B hidden-layer widths for a task name."""
    key = str(task).lower()
    if key not in _MLP_UNITS_BY_TASK:
        raise ValueError(
            f"Unknown task '{task}'. Known tasks: {sorted(_MLP_UNITS_BY_TASK)}"
        )
    return _MLP_UNITS_BY_TASK[key]
