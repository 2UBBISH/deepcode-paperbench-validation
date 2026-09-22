"""Per-stage output heads for the RoboticSequence (Meta-World) networks.

Appendix B.3 (SAC paragraph) of Wolczyk et al. (2024)::

    "We create a separate output head for each stage in the neural networks and
     then we use the stage ID information to choose the correct head. We found
     that this approach works better than adding the stage ID to the
     observation vector."

The stage ID is used ONLY to route the features to the right head; it is not
concatenated to the observation vector (in the main configuration).  This file
implements the head primitives that the SAC actor (:mod:`src.robotic_sequence.sac`)
uses:

* :class:`PolicyHead`     -- one Gaussian (mean + log-std) head for one stage.
* :class:`QHead`          -- one scalar Q head for one stage.
* :class:`PerStageHeadBank` -- bank of heads with stage-ID routing, used for both
  the policy and the Q-function (twin-Q uses two banks).

All heads are plain linear layers on top of the shared 4x256 MLP trunk.

The module is dependency-light: ``torch`` is imported defensively so that the
file can be imported (and statically analysed / unit tested) without PyTorch
installed.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

try:  # pragma: no cover - torch is expected in real runs
    import torch
    from torch import nn
    from torch.nn import functional as F

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _HAS_TORCH = False


__all__ = [
    "LOG_STD_MIN",
    "LOG_STD_MAX",
    "EPS",
    "PolicyHead",
    "QHead",
    "PerStageHeadBank",
    "make_policy_heads",
    "make_q_heads",
    "stage_index",
    "one_hot_stage",
]

#: Same clamping constants as ``src.robotic_sequence.sac`` (SAC's log-std range).
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
EPS = 1e-6


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise RuntimeError(
            "PyTorch is required for src.robotic_sequence.heads, but it could "
            "not be imported. Install torch to build RoboticSequence networks."
        )


def stage_index(
    stage_id: Union[int, Sequence[float], "torch.Tensor", None],
    n_stages: int,
    default: int = 0,
) -> Union[int, "torch.Tensor"]:
    """Normalise a stage identifier to an ``int`` or an ``int`` tensor.

    Accepts an integer index, a one-hot vector (list / tuple / tensor) or
    ``None``.  Values are always clamped to ``[0, n_stages - 1]`` so that a
    slightly off-by-one stage ID coming from the environment wrapper (which
    reports ``stage_id`` of the *next* stage after a success) can never crash a
    forward pass.
    """
    if n_stages <= 1:
        if torch is not None and isinstance(stage_id, torch.Tensor):
            return torch.zeros_like(stage_id, dtype=torch.long)
        return 0

    if stage_id is None:
        return default

    if isinstance(stage_id, torch.Tensor):
        t = stage_id
        if t.dim() == 0:
            return t.clamp(0, n_stages - 1).long()
        if t.dim() == 1:
            if t.numel() == n_stages and torch.is_floating_point(t):
                return t.argmax().clamp(0, n_stages - 1).long()
            return t.long().clamp(0, n_stages - 1)
        # (batch, n_stages) one-hot or (batch, 1) index
        if t.shape[-1] == n_stages and torch.is_floating_point(t):
            return t.argmax(dim=-1).clamp(0, n_stages - 1).long()
        return t.squeeze(-1).long().clamp(0, n_stages - 1)

    if isinstance(stage_id, (list, tuple)):
        if len(stage_id) == n_stages:
            best, best_v = 0, -float("inf")
            for i, v in enumerate(stage_id):
                if float(v) > best_v:
                    best, best_v = i, float(v)
            return best
        if len(stage_id) == 1:
            return int(stage_id[0])
        try:
            return max(0, min(n_stages - 1, int(stage_id[-1])))
        except (TypeError, ValueError):
            return default

    try:
        return max(0, min(n_stages - 1, int(stage_id)))
    except (TypeError, ValueError):
        return default


def one_hot_stage(
    stage_id: Union[int, Sequence[float], "torch.Tensor", None],
    n_stages: int,
    device: Any = None,
    dtype: Any = None,
) -> "torch.Tensor":
    """Return a one-hot encoding of ``stage_id`` of shape ``(n_stages,)``.

    Used when the stage ID *is* appended to the observation vector (the variant
    disabled by default in :mod:`src.robotic_sequence.env`).
    """
    _require_torch()
    if dtype is None:
        dtype = torch.get_default_dtype()
    if isinstance(stage_id, torch.Tensor) and stage_id.dim() >= 1:
        vec = stage_id.reshape(-1)[:n_stages]
        out = torch.zeros(n_stages, dtype=dtype, device=stage_id.device)
        idx = stage_index(stage_id, n_stages)
        out[int(idx) if not torch.is_tensor(idx) else int(idx.reshape(-1)[0])] = 1.0
        return out if device is None else out.to(device)
    idx = int(stage_index(stage_id, n_stages))  # type: ignore[arg-type]
    out = torch.zeros(n_stages, dtype=dtype, device=device)
    out[idx] = 1.0
    return out


class PolicyHead(nn.Module):
    """Gaussian policy head for a single stage: ``features -> (mean, log_std)``.

    Parameters
    ----------
    in_dim:
        Dimension of the shared trunk features.
    action_dim:
        Action dimensionality.
    log_std_min / log_std_max:
        Clamping range of the learned log standard deviation (SAC convention).
    init_log_std:
        Initial value of the (state-independent) log-std bias, ``-1`` by
        convention in SAC implementations.
    """

    def __init__(
        self,
        in_dim: int,
        action_dim: int,
        log_std_min: float = LOG_STD_MIN,
        log_std_max: float = LOG_STD_MAX,
        init_log_std: float = -1.0,
    ) -> None:
        _require_torch()
        super().__init__()
        self.in_dim = int(in_dim)
        self.action_dim = int(action_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        self.mean_layer = nn.Linear(self.in_dim, self.action_dim)
        self.log_std_layer = nn.Linear(self.in_dim, self.action_dim)
        self._init_weights(init_log_std)

    def _init_weights(self, init_log_std: float) -> None:
        bound = 1.0 / math.sqrt(max(1, self.in_dim))
        for layer in (self.mean_layer, self.log_std_layer):
            nn.init.uniform_(layer.weight, -bound, bound)
            nn.init.zeros_(layer.bias)
        if init_log_std is not None:
            with torch.no_grad():
                self.log_std_layer.bias.fill_(float(init_log_std))

    def forward(self, features: "torch.Tensor"):
        """Return ``(mean, log_std)`` each of shape ``(..., action_dim)``."""
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"in_dim={self.in_dim}, action_dim={self.action_dim}, "
            f"log_std=({self.log_std_min}, {self.log_std_max})"
        )


class QHead(nn.Module):
    """Scalar Q head for a single stage."""

    def __init__(self, in_dim: int, init_gain: float = 1.0) -> None:
        _require_torch()
        super().__init__()
        self.in_dim = int(in_dim)
        self.layer = nn.Linear(self.in_dim, 1)
        bound = init_gain / math.sqrt(max(1, self.in_dim))
        nn.init.uniform_(self.layer.weight, -bound, bound)
        nn.init.zeros_(self.layer.bias)

    def forward(self, features: "torch.Tensor") -> "torch.Tensor":
        """Return Q values of shape ``(..., 1)``."""
        return self.layer(features)


class PerStageHeadBank(nn.Module):
    """Bank of one head per stage, routed by the stage ID (Appendix B.3).

    ``forward(features, stage_id)`` evaluates the head selected by ``stage_id``
    for every sample in the batch, supporting:

    * a scalar index / one-hot vector shared by the whole batch,
    * a per-sample integer tensor of shape ``(batch,)``,
    * ``None`` (falls back to stage 0 when ``n_stages > 1``).

    When ``n_stages == 1`` the bank degenerates to exactly one head and the stage
    ID is ignored.
    """

    def __init__(
        self,
        in_dim: int,
        n_stages: int,
        head_factory,
        name: str = "heads",
    ) -> None:
        _require_torch()
        super().__init__()
        self.in_dim = int(in_dim)
        self.n_stages = max(1, int(n_stages))
        self.name = name
        self.heads = nn.ModuleList(
            [head_factory(self.in_dim) for _ in range(self.n_stages)]
        )

    def head(self, index: int) -> "nn.Module":
        return self.heads[max(0, min(self.n_stages - 1, int(index)))]

    def forward(self, features: "torch.Tensor", stage_id: Any = None):
        """Route ``features`` through the head(s) chosen by ``stage_id``.

        Returns a single head output when all samples share one stage (the common
        case) or a stacked tensor of shape ``(batch, ...)`` when the stages differ
        within the batch.
        """
        if self.n_stages == 1:
            return self.heads[0](features)

        idx = stage_index(stage_id, self.n_stages)
        is_tensor = torch.is_tensor(idx)
        if not is_tensor or idx.dim() == 0:
            index = int(idx) if not is_tensor else int(idx.item())
            return self.head(index)(features)

        idx = idx.reshape(-1)
        if idx.numel() == 1:
            return self.head(int(idx[0].item()))(features)

        # Mixed-stage batch: evaluate each head on its own subset of rows.
        outs: List[Any] = [None] * int(idx.numel())
        for s in range(self.n_stages):
            mask = idx == s
            if not bool(mask.any()):
                continue
            rows = features[mask]
            out = self.heads[s](rows)
            positions = mask.nonzero(as_tuple=True)[0]
            if isinstance(out, tuple):
                if outs[int(positions[0].item())] is None:
                    for pos in positions.tolist():
                        outs[pos] = [None] * len(out)
                for k in range(len(out)):
                    for j, pos in enumerate(positions.tolist()):
                        outs[pos][k] = out[k][j]  # type: ignore[index]
            else:
                for j, pos in enumerate(positions.tolist()):
                    outs[pos] = out[j]
        if isinstance(outs[0], tuple) or isinstance(outs[0], list):
            n_out = len(outs[0])  # type: ignore[arg-type]
            return tuple(
                torch.stack([outs[i][k] for i in range(len(outs))], dim=0)  # type: ignore[index]
                for k in range(n_out)
            )
        return torch.stack(outs, dim=0)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"in_dim={self.in_dim}, n_stages={self.n_stages}"


def make_policy_heads(
    in_dim: int,
    action_dim: int,
    n_stages: int = 1,
    log_std_min: float = LOG_STD_MIN,
    log_std_max: float = LOG_STD_MAX,
    init_log_std: float = -1.0,
) -> PerStageHeadBank:
    """Build a :class:`PerStageHeadBank` of Gaussian :class:`PolicyHead` objects."""

    def factory(dim: int) -> "PolicyHead":
        return PolicyHead(
            dim,
            action_dim,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            init_log_std=init_log_std,
        )

    return PerStageHeadBank(in_dim, n_stages, factory, name="policy_heads")


def make_q_heads(
    in_dim: int,
    n_stages: int = 1,
    init_gain: float = 1.0,
) -> PerStageHeadBank:
    """Build a :class:`PerStageHeadBank` of scalar :class:`QHead` objects."""

    def factory(dim: int) -> "QHead":
        return QHead(dim, init_gain=init_gain)

    return PerStageHeadBank(in_dim, n_stages, factory, name="q_heads")


def head_parameters(heads: Union["nn.Module", Mapping[str, "nn.Module"], None]):
    """Yield ``(name, parameter)`` pairs of a head or a bank (helper for EWC/BC)."""
    if heads is None:
        return
    if isinstance(heads, Mapping):
        for key, module in heads.items():
            for name, param in module.named_parameters():
                yield f"{key}.{name}", param
        return
    for name, param in heads.named_parameters():
        yield name, param
