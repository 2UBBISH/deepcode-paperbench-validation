"""FRE reward decoder ``q_theta(eta(s^d) | s^d, z)``.

Paper reference (Section 4.1, "Functional Reward Encoding", verbatim)::

    Training an FRE requires two neural networks,
    Encoder: p_theta(z | s^e_1, eta(s^e_1), s^e_2, eta(s^e_2), ..., s^e_K, eta(s^e_K)),
    Decoder: q_theta(eta(s^d) | s^d, z).

    "The decoder q_theta(eta(s) | s, z) is implemented as a feedforward neural
     network. Crucially, the states sampled for decoding are different than those
     used for encoding. The encoding network makes use of the entire set of
     (s_{1..K}, eta(s_{1..K})) pairs, whereas the decoder independently predicts
     the reward for each state, given the shared latent encoding z. We train both
     the encoder and decoder networks jointly, minimizing mean-squared error
     between the predicted and true rewards under the decoding states."

The reconstruction term of the variational lower bound of Equation (6) is

    E_{eta, L^e_eta, L^d_eta, z ~ p_theta(z | L^e_eta)}[
        sum_{k=1}^{K'} log q_theta(eta(s^d_k) | s^d_k, z)
        - beta * D_KL(p_theta(z | L^e_eta) || u(z)) ],

where ``log q_theta(eta(s^d_k) | s^d_k, z)`` is normalized to the MSE by the
paper's implementation above (a unit-variance Gaussian likelihood is equal to the
squared error up to an additive constant; ``u(z)`` is the unit Gaussian).

Hyper-parameters (Appendix A, Table 3):

    Reward Pairs to Decode (K') = 8
    Decoder Network Layers       = [512, 512, 512]
    Batch Size                   = 512
    Optimizer                    = Adam, Learning Rate = 0.0001

Addendum ("Additional Details on the FRE architecture"):

    "There is no embedding step for the observation state passed to the decoder.
     The raw state and the z-vector are concatenated directly."
    "The latent embedding (z) is 128-dimensional."

So the decoder input is ``concat(s^d, z)`` (raw state, no learned state
embedding), each decoding state is scored *independently* with the same shared
``z``, and the network is a plain MLP with widths ``[512, 512, 512]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Paper constants (Appendix A, Table 3 / Section 4.1 / Addendum)
# ---------------------------------------------------------------------------
DEFAULT_DECODER_LAYERS: tuple = (512, 512, 512)
DEFAULT_DECODER_ACTIVATION = "relu"  # not specified in the paper
DEFAULT_LATENT_DIM = 128  # addendum: "The latent embedding (z) is 128-dimensional"
DEFAULT_REWARD_DIM = 1
DEFAULT_NUM_DECODER_STATES = 8  # Table 3: "Reward Pairs to Decode = 8"
# Not specified in the paper: log-std clamp / init used only when the optional
# Gaussian likelihood form is enabled.
DEFAULT_LOG_STD_INIT = 0.0
DEFAULT_LOG_STD_MIN = -5.0
DEFAULT_LOG_STD_MAX = 5.0
DEFAULT_STATE_STD_FLOOR = 1e-6


def _make_activation(name: str) -> nn.Module:
    """Activation module by name (the paper does not specify the non-linearity)."""
    key = str(name).lower()
    factories = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
        "leakyrelu": nn.LeakyReLU,
        "identity": nn.Identity,
        "none": nn.Identity,
        "linear": nn.Identity,
    }
    if key not in factories:
        raise ValueError(f"Unsupported activation {name!r}")
    return factories[key]()


def _reduce(err: torch.Tensor, reduction: str) -> torch.Tensor:
    key = str(reduction).lower()
    if key in ("none", "no", "raw"):
        return err
    if key in ("mean", "avg", "average"):
        return err.mean()
    if key in ("sum", "total"):
        return err.sum()
    raise ValueError(f"Unsupported reduction {reduction!r}")


def _as_float_tensor(x: Any, device: Optional[torch.device] = None) -> torch.Tensor:
    if not torch.is_tensor(x):
        x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    x = x.to(dtype=torch.float32)
    if device is not None:
        x = x.to(device)
    return x


def clamp_std(std: torch.Tensor, floor: float = DEFAULT_STATE_STD_FLOOR) -> torch.Tensor:
    """Floor a standard deviation to avoid division by zero."""
    return torch.clamp(std, min=float(floor))


@dataclass
class DecoderInputs:
    """Container for the decoder's inputs (kept for interface symmetry)."""

    states: torch.Tensor  # (B, K', state_dim)
    z: torch.Tensor  # (B, latent_dim)

    @property
    def batch_size(self) -> int:
        return int(self.states.shape[0])

    @property
    def num_states(self) -> int:
        return int(self.states.shape[1]) if self.states.dim() == 3 else 1


@dataclass
class DecoderOutput:
    """Container for the decoder's predictions.

    ``mean`` holds the predicted reward for every decoding state, shaped
    ``(B, K')`` for the paper's setting (``(N,)`` for flat inputs).  ``log_std``
    is only populated when the optional learned Gaussian noise is enabled.
    """

    mean: torch.Tensor
    log_std: Optional[torch.Tensor] = None
    _target: Optional[torch.Tensor] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Reconstruction losses (Equation 6)
    # ------------------------------------------------------------------
    def squared_error(self, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Per-element squared error ``(eta(s^d) - eta_hat(s^d))^2``."""
        return (self.mean - self._resolve_target(target)) ** 2

    def mse(
        self,
        target: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """MSE between predicted and true rewards (paper's decoder objective)."""
        return _reduce(self.squared_error(target), reduction)

    def negative_log_likelihood(
        self,
        target: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """``-log q_theta(eta(s^d) | s^d, z)``; equals the MSE up to constants.

        The paper's implementation minimizes MSE, which corresponds to a
        unit-variance Gaussian likelihood (``log_std = 0``), i.e. this is the
        exact reconstruction term of Equation (6)::

            0.5 * (mu - y)^2 / var + log_std
        """
        tgt = self._resolve_target(target)
        if self.log_std is None:
            nll = 0.5 * (self.mean - tgt) ** 2
        else:
            log_std = self.log_std
            if log_std.dim() != self.mean.dim():
                log_std = log_std.expand_as(self.mean)
            var = torch.exp(2.0 * log_std)
            nll = 0.5 * (self.mean - tgt) ** 2 / var + log_std
        return _reduce(nll, reduction)

    def log_prob(
        self,
        target: Optional[torch.Tensor] = None,
        reduction: str = "none",
    ) -> torch.Tensor:
        """``log q_theta(eta(s^d) | s^d, z)``, summed over the K' decoding states."""
        return -self.negative_log_likelihood(target, reduction=reduction)

    def sum_log_prob(self, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``sum_{k=1}^{K'} log q_theta(eta(s^d_k) | s^d_k, z)``, reduced per batch."""
        err = self.negative_log_likelihood(target, reduction="none")
        if err.dim() > 1:
            err = err.reshape(err.shape[0], -1).sum(dim=-1)
        return -err

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_target(self, target: Optional[torch.Tensor]) -> torch.Tensor:
        if target is None:
            if self._target is None:
                raise ValueError(
                    "No target rewards supplied to the decoder output; call "
                    "mse(target) / negative_log_likelihood(target) instead."
                )
            target = self._target
        target = _as_float_tensor(target, device=self.mean.device)
        return target.reshape(self.mean.shape)

    def detach(self) -> "DecoderOutput":
        return DecoderOutput(
            mean=self.mean.detach(),
            log_std=None if self.log_std is None else self.log_std.detach(),
        )


class RewardDecoder(nn.Module):
    """Feed-forward reward decoder ``q_theta(eta(s^d) | s^d, z)``.

    The raw decoding state ``s^d`` is concatenated directly with the shared latent
    task embedding ``z`` (no state embedding step, per the addendum) and mapped
    through the Table 3 MLP widths ``[512, 512, 512]`` to a scalar reward
    prediction per state.  Each decoding state is predicted independently given
    the same ``z``; the decoder parameters (and therefore ``z``'s influence on the
    prediction) are shared across all decoding states.

    Parameters
    ----------
    state_dim:
        Dimensionality of the raw observation state (decoder input side).
    latent_dim:
        Dimensionality of ``z`` (128 in the paper).
    hidden_dims:
        Hidden widths; Table 3 gives ``[512, 512, 512]``.
    activation:
        Non-linearity between hidden layers (not specified in the paper; ReLU).
    layer_norm:
        Optional LayerNorm after each hidden linear layer (not specified in the
        paper).
    output_clip:
        Optional symmetric clipping of the predicted reward (not specified in the
        paper; ``None`` by default so MSE is unconstrained).
    learn_log_std:
        If ``True``, additionally predicts a per-state log-standard-deviation
        (clamped).  The paper's MSE objective corresponds to the fixed-variance
        case, so this is ``False`` by default.
    state_norm / normalize_states:
        Optional per-dimension observation normalization with the supplied
        dataset statistics.  Off by default: the paper concatenates the raw state
        (addendum), and the FRE trainer uses raw states.
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int = DEFAULT_LATENT_DIM,
        hidden_dims: Sequence[int] = DEFAULT_DECODER_LAYERS,
        activation: str = DEFAULT_DECODER_ACTIVATION,
        layer_norm: bool = False,
        output_dim: int = DEFAULT_REWARD_DIM,
        output_clip: Optional[float] = None,
        learn_log_std: bool = False,
        log_std_init: float = DEFAULT_LOG_STD_INIT,
        log_std_min: float = DEFAULT_LOG_STD_MIN,
        log_std_max: float = DEFAULT_LOG_STD_MAX,
        state_norm: bool = False,
        state_mean: Optional[Sequence[float]] = None,
        state_std: Optional[Sequence[float]] = None,
        normalize_states: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.activation_name = str(activation)
        self.layer_norm = bool(layer_norm)
        self.output_dim = int(output_dim)
        self.output_clip = None if output_clip is None else float(output_clip)
        self.learn_log_std = bool(learn_log_std)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        if normalize_states is not None:
            state_norm = bool(normalize_states)
        if state_mean is not None:
            self.register_buffer(
                "state_mean",
                _as_float_tensor(state_mean).reshape(-1),
                persistent=False,
            )
        else:
            self.register_buffer("state_mean", None, persistent=False)
        if state_std is not None:
            self.register_buffer(
                "state_std",
                _as_float_tensor(state_std).reshape(-1),
                persistent=False,
            )
        else:
            self.register_buffer("state_std", None, persistent=False)
        self.state_norm = (
            bool(state_norm) and self.state_mean is not None and self.state_std is not None
        )

        # Input is the concatenation of the raw state and z (addendum: "There is no
        # embedding step for the observation state passed to the decoder. The raw
        # state and the z-vector are concatenated directly.").
        in_dim = self.state_dim + self.latent_dim
        layers: list = []
        prev = in_dim
        for hidden in self.hidden_dims:
            layers.append(nn.Linear(prev, hidden))
            if self.layer_norm:
                layers.append(nn.LayerNorm(hidden))
            layers.append(_make_activation(self.activation_name))
            prev = hidden
        layers.append(nn.Linear(prev, self.output_dim))
        self.net = nn.Sequential(*layers)

        if self.learn_log_std:
            self.log_std_head = nn.Linear(prev, self.output_dim)
            nn.init.constant_(self.log_std_head.bias, float(log_std_init))
        else:
            self.log_std_head = None

        self._input_dim = in_dim

    # ------------------------------------------------------------------
    # Properties / input handling
    # ------------------------------------------------------------------
    @property
    def input_dim(self) -> int:
        """Dimensionality of the concatenated ``(s^d, z)`` decoder input."""
        return self._input_dim

    @property
    def token_dim(self) -> int:
        """Alias of :attr:`input_dim`."""
        return self._input_dim

    def _normalize_states(self, states: torch.Tensor) -> torch.Tensor:
        if not self.state_norm or self.state_mean is None or self.state_std is None:
            return states
        mean = self.state_mean.to(states.device, states.dtype)
        std = clamp_std(self.state_std.to(states.device, states.dtype))
        return (states - mean) / std

    def _prepare_inputs(self, states: Any, z: Any) -> torch.Tensor:
        """Concatenate raw decoding states with the shared latent ``z``.

        Supports ``states`` of shape ``(B, K', state_dim)`` (the paper's
        per-state decoding setting) or ``(N, state_dim)`` (flat, ``N = B * K'``).
        ``z`` may be ``(B, latent_dim)``, ``(latent_dim,)`` or pre-expanded
        ``(B, K', latent_dim)``.
        """
        s = _as_float_tensor(states)
        if torch.is_tensor(z):
            zt = z.to(dtype=torch.float32, device=s.device)
        else:
            zt = _as_float_tensor(z, device=s.device)

        if s.dim() == 1:
            s = s.unsqueeze(0)
        if s.dim() not in (2, 3):
            raise ValueError(f"Unsupported state shape {tuple(s.shape)}")

        if s.dim() == 2:
            n = s.shape[0]
            if zt.dim() == 1:
                zt = zt.unsqueeze(0).expand(n, -1)
            elif zt.dim() == 3:
                # (B, K', L) flattened to match the flattened states.
                zt = zt.reshape(-1, zt.shape[-1])
            if zt.shape[0] != n:
                if zt.shape[0] == 1:
                    zt = zt.expand(n, -1)
                else:
                    raise ValueError(
                        f"Cannot align states {tuple(s.shape)} with latent z "
                        f"{tuple(zt.shape)}"
                    )
            return torch.cat([s, zt], dim=-1)

        b, k, _ = s.shape
        if zt.dim() == 1:
            zt = zt.unsqueeze(0)
        if zt.dim() == 2:
            if zt.shape[0] != b:
                if zt.shape[0] == 1:
                    zt = zt.expand(b, -1)
                else:
                    raise ValueError(
                        f"Cannot align states {tuple(s.shape)} with latent z "
                        f"{tuple(zt.shape)}"
                    )
            zt = zt.unsqueeze(1).expand(b, k, zt.shape[-1])
        elif zt.dim() == 3:
            if zt.shape[0] != b or zt.shape[1] not in (1, k):
                raise ValueError(
                    f"Cannot align states {tuple(s.shape)} with latent z "
                    f"{tuple(zt.shape)}"
                )
            if zt.shape[1] == 1:
                zt = zt.expand(b, k, zt.shape[-1])
        else:
            raise ValueError(f"Unsupported latent shape {tuple(zt.shape)}")
        return torch.cat([s, zt], dim=-1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        states: Any,
        z: Any,
        target_rewards: Optional[Any] = None,
        return_output: bool = False,
    ) -> Union[torch.Tensor, DecoderOutput]:
        """Predict ``eta(s^d)`` for each decoding state given the shared ``z``.

        Returns the reward prediction with shape ``(B, K')`` (or ``(N,)`` for flat
        inputs).  If ``return_output=True`` (or a target is provided) a
        :class:`DecoderOutput` is returned instead, carrying the optional
        ``log_std`` and ready-to-use MSE / NLL helpers.
        """
        x = self._prepare_inputs(states, z)
        x = self._normalize_states(x)
        mean = self.net(x)
        if self.output_clip is not None:
            mean = torch.clamp(mean, -self.output_clip, self.output_clip)

        log_std: Optional[torch.Tensor] = None
        if self.learn_log_std and self.log_std_head is not None:
            log_std = self.log_std_head(x)
            if log_std.dim() == 3 and log_std.shape[-1] != 1:
                log_std = log_std.mean(dim=-1, keepdim=True)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        # Squeeze the trailing scalar-reward dimension: (B, K', 1) -> (B, K').
        if mean.dim() == 3:
            if mean.shape[-1] == 1:
                mean = mean.squeeze(-1)
            else:
                mean = mean.mean(dim=-1)
        if log_std is not None:
            if log_std.dim() == mean.dim() + 1 and log_std.shape[-1] == 1:
                log_std = log_std.squeeze(-1)
            elif log_std.shape != mean.shape:
                log_std = log_std.reshape(mean.shape)

        if return_output or target_rewards is not None:
            return DecoderOutput(mean=mean, log_std=log_std, _target=target_rewards)
        return mean

    def predict(
        self,
        states: Any,
        z: Any,
        target_rewards: Optional[Any] = None,
    ) -> DecoderOutput:
        """Forward pass returning a :class:`DecoderOutput`."""
        out = self.forward(states, z, target_rewards=target_rewards, return_output=True)
        if not isinstance(out, DecoderOutput):  # pragma: no cover - defensive
            raise RuntimeError("Expected DecoderOutput from forward()")
        return out

    # ------------------------------------------------------------------
    # Losses: reconstruction term of Equation (6)
    # ------------------------------------------------------------------
    def reconstruction_loss(
        self,
        states: Any,
        z: Any,
        target_rewards: Any,
        loss_type: str = "mse",
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Reconstruction loss on the decoding states.

        The paper minimizes the mean-squared error between the predicted and true
        rewards; this returns the negative of the corresponding (per-batch) term
        ``sum_{k=1}^{K'} log q_theta(eta(s^d_k) | s^d_k, z)``.
        """
        out = self.predict(states, z, target_rewards=target_rewards)
        key = str(loss_type).lower()
        if key in ("mse", "l2", "squared", "squared_error"):
            return out.mse(reduction=reduction)
        if key in ("nll", "gaussian", "negative_log_likelihood", "log_likelihood"):
            return out.negative_log_likelihood(reduction=reduction)
        if key in ("sum_nll", "sum_negative_log_likelihood", "neg_sum_log_prob"):
            return _reduce(
                out.negative_log_likelihood(reduction="none"), reduction
            )
        raise ValueError(f"Unsupported reconstruction loss {loss_type!r}")

    def mse_loss(
        self,
        states: Any,
        z: Any,
        target_rewards: Any,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """MSE reconstruction loss (the paper's decoder objective)."""
        return self.reconstruction_loss(states, z, target_rewards, "mse", reduction)

    def nll_loss(
        self,
        states: Any,
        z: Any,
        target_rewards: Any,
        reduction: str = "sum",
    ) -> torch.Tensor:
        """Gaussian NLL (equals MSE up to a constant for the fixed-variance case)."""
        return self.reconstruction_loss(states, z, target_rewards, "nll", reduction)

    @torch.no_grad()
    def decode_states(self, states: Any, z: Any) -> torch.Tensor:
        """Deterministic reward prediction for evaluation/logging."""
        out = self.forward(states, z)
        assert torch.is_tensor(out)
        return out

    @torch.no_grad()
    def predict_numpy(self, states: Any, z: Any) -> np.ndarray:
        """NumPy-friendly wrapper used by evaluation and diagnostics."""
        return self.decode_states(states, z).detach().cpu().numpy()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def describe(self) -> Dict[str, Any]:
        """Human-readable summary of the decoder configuration."""
        return {
            "name": "fre_reward_decoder",
            "state_dim": self.state_dim,
            "latent_dim": self.latent_dim,
            "hidden_dims": list(self.hidden_dims),
            "input_dim": self.input_dim,
            "activation": self.activation_name,
            "layer_norm": self.layer_norm,
            "output_dim": self.output_dim,
            "output_clip": self.output_clip,
            "learn_log_std": self.learn_log_std,
            "num_decoder_states": DEFAULT_NUM_DECODER_STATES,
            "num_parameters": int(sum(p.numel() for p in self.parameters())),
        }


def make_reward_decoder(
    state_dim: int,
    latent_dim: int = DEFAULT_LATENT_DIM,
    hidden_dims: Sequence[int] = DEFAULT_DECODER_LAYERS,
    **kwargs: Any,
) -> RewardDecoder:
    """Factory mirroring :func:`fre.models.encoder.make_fre_encoder`.

    Uses the paper's Table 3 decoder widths ``[512, 512, 512]`` and the 128-d
    latent ``z`` by default.
    """
    return RewardDecoder(
        state_dim=state_dim,
        latent_dim=latent_dim,
        hidden_dims=tuple(hidden_dims),
        **kwargs,
    )


__all__ = [
    "DEFAULT_DECODER_LAYERS",
    "DEFAULT_DECODER_ACTIVATION",
    "DEFAULT_LATENT_DIM",
    "DEFAULT_REWARD_DIM",
    "DEFAULT_NUM_DECODER_STATES",
    "DEFAULT_LOG_STD_INIT",
    "DEFAULT_LOG_STD_MIN",
    "DEFAULT_LOG_STD_MAX",
    "DecoderInputs",
    "DecoderOutput",
    "RewardDecoder",
    "make_reward_decoder",
    "clamp_std",
]
