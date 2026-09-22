"""Network definitions and model-assembly utilities for RoboticSequence (Meta-World) SAC.

This module is the "model" layer of the RoboticSequence track.  It defines the
architecture required by Appendix B.3 of Wolczyk et al. (2024) and used for every
Meta-World experiment:

* a **4-layer MLP with 256 hidden units** (``hidden_dim=256``, ``num_hidden_layers=4``),
* **Leaky-ReLU** activations (negative slope ``0.01``),
* **LayerNorm after the first hidden layer**,
* **one output head per stage**, selected by the stage-ID one-hot,
* a **twin (clipped-double) Q critic** with its own per-stage heads,
* **automatic entropy-coefficient tuning** (the learnable ``alpha`` lives in
  :mod:`src.robotic_sequence.sac`; here we only build the networks),
* and **Adam** optimisation at ``lr=1e-3`` with ``batch_size=128``.

The heavy lifting (trunk/head/policy/critic implementations) lives in
:mod:`src.robotic_sequence.sac` and :mod:`src.robotic_sequence.heads`; this file
assembles those primitives into actor/critic pairs, keeps target networks in sync
via Polyak averaging (``tau=0.005``), and exposes the pieces the analysis modules
need (feature extraction and layer names for CKA, parameter lists for the
actor-only retention losses EWC / BC / EM).

Everything torch-related is imported defensively so this module stays importable —
and statically analysable — in a torch-less environment.  Building or running a
model requires :mod:`torch` and raises a clear ``RuntimeError`` otherwise.

Reference: Appendix B.3 (architecture / hyperparameters) of
"Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation
Problem" (Wolczyk et al., 2024).  SAC itself follows
https://spinningup.openai.com/en/latest/algorithms/sac.html.
"""

from __future__ import annotations

import argparse
import inspect
import math
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "RoboticModelConfig",
    "build_actor",
    "build_critic",
    "build_models",
    "build_agent",
    "build_sac_models",
    "ActorCritic",
    "TargetNetwork",
    "soft_update",
    "make_optimizers",
    "policy_features",
    "q_features",
    "layer_names",
    "analyzer_layers",
    "actor_parameters",
    "actor_head_parameters",
    "parameter_groups",
    "count_parameters",
    "save_model",
    "load_model",
    "describe",
    "MLPTrunk",
    "PerStageHeads",
    "SACPolicy",
    "SACTwinQ",
    "SquashedNormal",
    "HIDDEN_DIM",
    "NUM_HIDDEN_LAYERS",
    "ACTIVATION",
    "LEAKY_RELU_SLOPE",
    "LAYER_NORM_AFTER_FIRST",
    "DEFAULT_LR",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_TAU",
    "DEFAULT_GAMMA",
    "MODEL_DEFAULTS",
    "main",
]

# --------------------------------------------------------------------------------------
# Optional torch import
# --------------------------------------------------------------------------------------
try:  # pragma: no cover - exercised only when torch is missing
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _HAS_TORCH = False


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise RuntimeError(
            "PyTorch is required to build/run RoboticSequence models "
            "(pip install torch)."
        )


# --------------------------------------------------------------------------------------
# Paper constants (Appendix B.3)
# --------------------------------------------------------------------------------------
HIDDEN_DIM = 256
NUM_HIDDEN_LAYERS = 4
ACTIVATION = "leaky_relu"
LEAKY_RELU_SLOPE = 0.01
LAYER_NORM_AFTER_FIRST = True
DEFAULT_LR = 1e-3
DEFAULT_BATCH_SIZE = 128
DEFAULT_TAU = 0.005
DEFAULT_GAMMA = 0.99
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0

MODEL_DEFAULTS: Dict[str, Any] = {
    "hidden_dim": HIDDEN_DIM,
    "num_hidden_layers": NUM_HIDDEN_LAYERS,
    "activation": ACTIVATION,
    "leaky_relu_slope": LEAKY_RELU_SLOPE,
    "layer_norm_after_first": LAYER_NORM_AFTER_FIRST,
    "per_stage_heads": True,
    "use_twin_q": True,
    "stage_id_in_obs": False,
    "log_std_min": LOG_STD_MIN,
    "log_std_max": LOG_STD_MAX,
    "lr": DEFAULT_LR,
    "batch_size": DEFAULT_BATCH_SIZE,
    "tau": DEFAULT_TAU,
    "gamma": DEFAULT_GAMMA,
    "weight_decay": 0.0,
    "device": "cpu",
    "seed": None,
}


# --------------------------------------------------------------------------------------
# Primitives: imported from the sibling modules when available, otherwise fallback stubs
# --------------------------------------------------------------------------------------
def _import_from_sac(name: str, default: Any = None) -> Any:
    """Best-effort import of a symbol from :mod:`src.robotic_sequence.sac`."""
    for module_name in ("src.robotic_sequence.sac", "sac"):
        try:
            module = __import__(module_name, fromlist=[name])
        except Exception:  # pragma: no cover - torch-less or partial installs
            continue
        value = getattr(module, name, None)
        if value is not None:
            return value
    return default


def _import_from_heads(name: str, default: Any = None) -> Any:
    for module_name in ("src.robotic_sequence.heads", "heads"):
        try:
            module = __import__(module_name, fromlist=[name])
        except Exception:  # pragma: no cover
            continue
        value = getattr(module, name, None)
        if value is not None:
            return value
    return default


MLPTrunk = _import_from_sac("MLPTrunk")
PerStageHeads = _import_from_sac("PerStageHeads")
SACPolicy = _import_from_sac("SACPolicy")
SACTwinQ = _import_from_sac("SACTwinQ")
SquashedNormal = _import_from_sac("SquashedNormal")

_make_policy_heads = _import_from_heads("make_policy_heads")
_make_q_heads = _import_from_heads("make_q_heads")
_head_parameters = _import_from_heads("head_parameters")


if _HAS_TORCH and MLPTrunk is None:  # pragma: no cover - only without sac.py

    def _activation_module(name: str, slope: float = LEAKY_RELU_SLOPE) -> Any:
        name = (name or "leaky_relu").lower()
        if name in ("relu",):
            return nn.ReLU()
        if name in ("gelu",):
            return nn.GELU()
        if name in ("tanh",):
            return nn.Tanh()
        return nn.LeakyReLU(slope)

    class MLPTrunk(nn.Module):  # type: ignore[no-redef]
        """4x256 MLP trunk with Leaky-ReLU and LayerNorm after the first layer."""

        def __init__(
            self,
            in_dim: int,
            hidden_dim: int = HIDDEN_DIM,
            num_hidden_layers: int = NUM_HIDDEN_LAYERS,
            activation: str = ACTIVATION,
            leaky_relu_slope: float = LEAKY_RELU_SLOPE,
            layer_norm_after_first: bool = True,
            **_: Any,
        ) -> None:
            super().__init__()
            layers: List[nn.Module] = []
            last = int(in_dim)
            for index in range(int(num_hidden_layers)):
                layers.append(nn.Linear(last, int(hidden_dim)))
                if index == 0 and layer_norm_after_first:
                    layers.append(nn.LayerNorm(int(hidden_dim)))
                layers.append(_activation_module(activation, leaky_relu_slope))
                last = int(hidden_dim)
            self.net = nn.Sequential(*layers)
            self.output_dim = int(hidden_dim)

        def forward(self, x: Any) -> Any:
            return self.net(x)

    class PerStageHeads(nn.Module):  # type: ignore[no-redef]
        """One linear head per stage, selected by the (one-hot) stage ID."""

        def __init__(self, in_dim: int, out_dim: int, n_stages: int = 1, **_: Any) -> None:
            super().__init__()
            self.n_stages = max(1, int(n_stages))
            self.out_dim = int(out_dim)
            self.heads = nn.ModuleList(
                [nn.Linear(int(in_dim), int(out_dim)) for _ in range(self.n_stages)]
            )

        @staticmethod
        def stage_index(stage_id: Any, n_stages: int) -> Any:
            if stage_id is None:
                return 0
            if torch.is_tensor(stage_id):
                if stage_id.dim() == 0:
                    return int(stage_id.item())
                if stage_id.dim() == 1:
                    if stage_id.numel() == 1:
                        return int(stage_id.item())
                    return int(stage_id.reshape(-1).argmax().item())
                return stage_id
            if isinstance(stage_id, (list, tuple)):
                return int(max(range(len(stage_id)), key=lambda i: stage_id[i]))
            return int(stage_id)

        def forward(self, features: Any, stage_id: Any = None) -> Any:
            index = self.stage_index(stage_id, self.n_stages)
            if torch.is_tensor(index) and index.dim() == 1:
                per_sample = [
                    self.heads[min(max(int(i), 0), self.n_stages - 1)](features)
                    for i in index.tolist()
                ]
                return torch.stack(per_sample, dim=0)
            index = min(max(int(index), 0), self.n_stages - 1)
            return self.heads[index](features)


if _HAS_TORCH and SquashedNormal is None:  # pragma: no cover - only without sac.py

    class SquashedNormal:  # type: ignore[no-redef]
        """Minimal tanh-squashed normal (used only when sac.py is unavailable)."""

        def __init__(self, mean: Any = None, log_std: Any = None, dist: Any = None) -> None:
            base = dist if dist is not None else torch.distributions.Normal(mean, log_std.exp())
            self.dist = base
            self.mean = base.mean
            self.std = base.stddev

        @property
        def loc(self) -> Any:
            return self.dist.mean

        @property
        def scale(self) -> Any:
            return self.dist.stddev

        @property
        def mean_action(self) -> Any:
            return torch.tanh(self.dist.mean)

        def sample(self, sample_shape: Any = torch.Size()) -> Any:
            return torch.tanh(self.dist.rsample(sample_shape))

        def rsample(self, sample_shape: Any = torch.Size()) -> Any:
            return torch.tanh(self.dist.rsample(sample_shape))

        def log_prob(self, action: Any) -> Any:
            action = torch.clamp(action, -1.0 + 1e-6, 1.0 - 1e-6)
            pre = torch.atanh(action)
            correction = torch.log(torch.clamp(1.0 - action.pow(2), min=1e-6))
            return self.dist.log_prob(pre) - correction

        def entropy(self) -> Any:
            return self.dist.entropy()


if _HAS_TORCH and SACPolicy is None:  # pragma: no cover - only without sac.py

    class SACPolicy(nn.Module):  # type: ignore[no-redef]
        """Gaussian policy with a shared trunk and per-stage (mean, log_std) heads."""

        def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            n_stages: int = 1,
            hidden_dim: int = HIDDEN_DIM,
            num_hidden_layers: int = NUM_HIDDEN_LAYERS,
            activation: str = ACTIVATION,
            leaky_relu_slope: float = LEAKY_RELU_SLOPE,
            layer_norm_after_first: bool = True,
            per_stage_heads: bool = True,
            stage_id_in_obs: bool = False,
            log_std_min: float = LOG_STD_MIN,
            log_std_max: float = LOG_STD_MAX,
            **_: Any,
        ) -> None:
            super().__init__()
            self.obs_dim = int(obs_dim)
            self.action_dim = int(action_dim)
            self.n_stages = max(1, int(n_stages))
            self.stage_id_in_obs = bool(stage_id_in_obs)
            self.log_std_min = float(log_std_min)
            self.log_std_max = float(log_std_max)
            trunk_in = self.obs_dim + (self.n_stages if self.stage_id_in_obs else 0)
            self.trunk = MLPTrunk(
                trunk_in,
                hidden_dim,
                num_hidden_layers,
                activation,
                leaky_relu_slope,
                layer_norm_after_first,
            )
            self.heads = PerStageHeads(self.trunk.output_dim, 2 * self.action_dim, self.n_stages)

        def features(self, obs: Any, stage_id: Any = None) -> Any:
            x = obs
            if self.stage_id_in_obs:
                onehot = _one_hot_stage(stage_id, self.n_stages, x)
                x = torch.cat([x, onehot], dim=-1)
            return self.trunk(x)

        def distribution(self, obs: Any, stage_id: Any = None) -> Any:
            out = self.heads(self.features(obs, stage_id), stage_id)
            mean, log_std = torch.split(out, self.action_dim, dim=-1)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            return SquashedNormal(mean, log_std)

        def forward(self, obs: Any, stage_id: Any = None) -> Any:
            return self.distribution(obs, stage_id)

        def log_prob(self, obs: Any, actions: Any = None, stage_id: Any = None, **kw: Any) -> Any:
            dist = self.distribution(obs, stage_id)
            if actions is None:
                return dist.sample(), dist.log_prob(dist.mean_action)
            return dist.log_prob(actions)

        def act(self, obs: Any, deterministic: bool = False, stage_id: Any = None, **kw: Any) -> Any:
            dist = self.distribution(obs, stage_id)
            if deterministic:
                return dist.mean_action
            return dist.sample()

        def get_actions(self, obs: Any, stage_id: Any = None, deterministic: bool = False) -> Any:
            return self.act(obs, deterministic=deterministic, stage_id=stage_id)


if _HAS_TORCH and SACTwinQ is None:  # pragma: no cover - only without sac.py

    class SACTwinQ(nn.Module):  # type: ignore[no-redef]
        """Twin Q-networks with per-stage output heads."""

        def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            n_stages: int = 1,
            hidden_dim: int = HIDDEN_DIM,
            num_hidden_layers: int = NUM_HIDDEN_LAYERS,
            activation: str = ACTIVATION,
            leaky_relu_slope: float = LEAKY_RELU_SLOPE,
            layer_norm_after_first: bool = True,
            per_stage_heads: bool = True,
            use_twin_q: bool = True,
            **_: Any,
        ) -> None:
            super().__init__()
            self.obs_dim = int(obs_dim)
            self.action_dim = int(action_dim)
            self.n_stages = max(1, int(n_stages))
            in_dim = self.obs_dim + self.action_dim
            self.q1 = MLPTrunk(
                in_dim, hidden_dim, num_hidden_layers, activation,
                leaky_relu_slope, layer_norm_after_first,
            )
            self.q2 = MLPTrunk(
                in_dim, hidden_dim, num_hidden_layers, activation,
                leaky_relu_slope, layer_norm_after_first,
            )
            self.q1_heads = PerStageHeads(self.q1.output_dim, 1, self.n_stages)
            self.q2_heads = PerStageHeads(self.q2.output_dim, 1, self.n_stages)

        def forward(self, obs: Any, action: Any, stage_id: Any = None) -> Tuple[Any, Any]:
            x = torch.cat([obs, action], dim=-1)
            q1 = self.q1_heads(self.q1(x), stage_id).squeeze(-1)
            q2 = self.q2_heads(self.q2(x), stage_id).squeeze(-1)
            return q1, q2

        def q1_value(self, obs: Any, action: Any, stage_id: Any = None) -> Any:
            x = torch.cat([obs, action], dim=-1)
            return self.q1_heads(self.q1(x), stage_id).squeeze(-1)


def _one_hot_stage(stage_id: Any, n_stages: int, reference: Any) -> Any:
    """Build a one-hot stage encoding matching ``reference``'s leading dimensions."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch required")
    batch = int(reference.shape[0]) if reference.dim() >= 1 else 1
    device = reference.device
    dtype = reference.dtype
    out = torch.zeros(batch, n_stages, device=device, dtype=dtype)
    if stage_id is None:
        return out
    if torch.is_tensor(stage_id):
        if stage_id.dim() == 0:
            out[:, int(stage_id.item()) % n_stages] = 1.0
        elif stage_id.dim() == 1 and stage_id.numel() == n_stages:
            out = stage_id.reshape(1, -1).expand(batch, -1).to(device=device, dtype=dtype)
        else:
            ids = stage_id.reshape(-1).long() % n_stages
            out = torch.zeros(batch, n_stages, device=device, dtype=dtype)
            out[torch.arange(min(batch, ids.numel()), device=device), ids[:batch]] = 1.0
        return out
    if isinstance(stage_id, (list, tuple)):
        out = torch.zeros(batch, n_stages, device=device, dtype=dtype)
        out[:, int(max(range(len(stage_id)), key=lambda i: stage_id[i])) % n_stages] = 1.0
        return out
    out[:, int(stage_id) % n_stages] = 1.0
    return out


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
def _cget(obj: Any, key: str, default: Any = None) -> Any:
    """Tolerant nested getter supporting dotted paths over dicts/objects."""
    if obj is None:
        return default
    current = obj
    for part in key.split("."):
        if current is None:
            return default
        if isinstance(current, Mapping):
            current = current.get(part, None)
        elif isinstance(current, (list, tuple)):
            try:
                current = current[int(part)]
            except Exception:
                return default
        else:
            current = getattr(current, part, None)
    return default if current is None else current


@dataclass
class RoboticModelConfig:
    """Architecture + optimisation hyperparameters for RoboticSequence models.

    Defaults are exactly Appendix B.3 / Table 3 of the paper: a 4-layer MLP with
    256 hidden units, Leaky-ReLU, LayerNorm after the first layer, one output head
    per stage, Adam with ``lr=1e-3``, ``batch_size=128``, ``tau=0.005``,
    ``gamma=0.99``.
    """

    hidden_dim: int = HIDDEN_DIM
    num_hidden_layers: int = NUM_HIDDEN_LAYERS
    activation: str = ACTIVATION
    leaky_relu_slope: float = LEAKY_RELU_SLOPE
    layer_norm_after_first: bool = LAYER_NORM_AFTER_FIRST
    per_stage_heads: bool = True
    use_twin_q: bool = True
    stage_id_in_obs: bool = False
    log_std_min: float = LOG_STD_MIN
    log_std_max: float = LOG_STD_MAX
    lr: float = DEFAULT_LR
    batch_size: int = DEFAULT_BATCH_SIZE
    tau: float = DEFAULT_TAU
    gamma: float = DEFAULT_GAMMA
    weight_decay: float = 0.0
    device: str = "cpu"
    seed: Optional[int] = None

    # ---- helpers --------------------------------------------------------------------
    def with_overrides(self, **overrides: Any) -> "RoboticModelConfig":
        data = asdict(self)
        for key, value in overrides.items():
            if value is not None and key in data:
                data[key] = value
        return RoboticModelConfig(**data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def hidden_sizes(self) -> Tuple[int, ...]:
        return tuple(int(self.hidden_dim) for _ in range(int(self.num_hidden_layers)))

    @classmethod
    def from_config(cls, cfg: Any = None, **overrides: Any) -> "RoboticModelConfig":
        """Build from a ``configs/robotic_sequence.yaml``-style config object.

        Reads the ``sac`` / ``model`` / ``env`` / ``finetune`` blocks tolerantly so
        callers can pass either a ``Config``, a plain mapping, or a dataclass.
        """
        values: Dict[str, Any] = dict(MODEL_DEFAULTS)

        def pick(*paths: str, default: Any = None) -> Any:
            for path in paths:
                found = _cget(cfg, path, None)
                if found is not None:
                    return found
            return default

        mapping = {
            "hidden_dim": ("sac.hidden_dim", "model.hidden_dim", "hidden_dim"),
            "num_hidden_layers": ("sac.num_hidden_layers", "model.num_hidden_layers"),
            "activation": ("sac.activation", "model.activation"),
            "leaky_relu_slope": ("sac.leaky_relu_slope", "model.leaky_relu_slope"),
            "layer_norm_after_first": (
                "sac.layer_norm_after_first", "model.layer_norm_after_first",
            ),
            "per_stage_heads": ("sac.per_stage_heads", "model.per_stage_heads"),
            "use_twin_q": ("sac.use_twin_q", "model.use_twin_q"),
            "stage_id_in_obs": ("env.append_stage_onehot", "model.stage_id_in_obs"),
            "lr": ("sac.lr", "model.lr", "finetune.lr"),
            "batch_size": ("sac.batch_size", "model.batch_size", "finetune.batch_size"),
            "tau": ("sac.tau", "model.tau"),
            "gamma": ("sac.gamma", "model.gamma"),
            "weight_decay": ("sac.weight_decay", "model.weight_decay"),
            "device": ("compute.device", "device"),
            "seed": ("seed",),
        }
        for target, paths in mapping.items():
            found = pick(*paths, default=None)
            if found is not None:
                values[target] = found

        for key, value in overrides.items():
            if value is not None and key in values:
                values[key] = value
        return cls(**values)  # type: ignore[arg-type]


def _filtered_kwargs(factory: Any, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep only kwargs accepted by ``factory`` (robust to signature drift)."""
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):  # pragma: no cover
        return dict(kwargs)
    if any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values()):
        return dict(kwargs)
    allowed = set(signature.parameters)
    return {k: v for k, v in kwargs.items() if k in allowed}


# --------------------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------------------
def build_actor(
    cfg: Any = None,
    obs_dim: int = 0,
    action_dim: int = 0,
    n_stages: int = 1,
    **overrides: Any,
) -> Any:
    """Build a Gaussian actor with per-stage heads (Appendix B.3).

    ``cfg`` may be a :class:`RoboticModelConfig`, a YAML-style config, ``None``, or
    a raw ``sac`` mapping.
    """
    _require_torch()
    if isinstance(cfg, RoboticModelConfig):
        model_cfg = cfg.with_overrides(**overrides)
    else:
        model_cfg = RoboticModelConfig.from_config(cfg, **overrides)

    kwargs = _filtered_kwargs(
        SACPolicy,
        dict(
            obs_dim=int(obs_dim),
            action_dim=int(action_dim),
            n_stages=int(n_stages),
            hidden_dim=int(model_cfg.hidden_dim),
            num_hidden_layers=int(model_cfg.num_hidden_layers),
            activation=model_cfg.activation,
            leaky_relu_slope=float(model_cfg.leaky_relu_slope),
            layer_norm_after_first=bool(model_cfg.layer_norm_after_first),
            per_stage_heads=bool(model_cfg.per_stage_heads),
            stage_id_in_obs=bool(model_cfg.stage_id_in_obs),
            log_std_min=float(model_cfg.log_std_min),
            log_std_max=float(model_cfg.log_std_max),
        ),
    )
    policy = SACPolicy(**kwargs)
    policy.to(model_cfg.device)
    return policy


def build_critic(
    cfg: Any = None,
    obs_dim: int = 0,
    action_dim: int = 0,
    n_stages: int = 1,
    **overrides: Any,
) -> Any:
    """Build the (twin) Q critic with per-stage heads."""
    _require_torch()
    if isinstance(cfg, RoboticModelConfig):
        model_cfg = cfg.with_overrides(**overrides)
    else:
        model_cfg = RoboticModelConfig.from_config(cfg, **overrides)

    kwargs = _filtered_kwargs(
        SACTwinQ,
        dict(
            obs_dim=int(obs_dim),
            action_dim=int(action_dim),
            n_stages=int(n_stages),
            hidden_dim=int(model_cfg.hidden_dim),
            num_hidden_layers=int(model_cfg.num_hidden_layers),
            activation=model_cfg.activation,
            leaky_relu_slope=float(model_cfg.leaky_relu_slope),
            layer_norm_after_first=bool(model_cfg.layer_norm_after_first),
            per_stage_heads=bool(model_cfg.per_stage_heads),
            use_twin_q=bool(model_cfg.use_twin_q),
        ),
    )
    critic = SACTwinQ(**kwargs)
    critic.to(model_cfg.device)
    return critic


def build_models(
    cfg: Any = None,
    obs_dim: int = 0,
    action_dim: int = 0,
    n_stages: int = 1,
    device: Optional[str] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Build ``{"actor", "critic", "config", "device"}`` for SAC."""
    _require_torch()
    if isinstance(cfg, RoboticModelConfig):
        model_cfg = cfg.with_overrides(**overrides)
    else:
        model_cfg = RoboticModelConfig.from_config(cfg, **overrides)
    if device is not None:
        model_cfg = model_cfg.with_overrides(device=device)

    actor = build_actor(model_cfg, obs_dim, action_dim, n_stages)
    critic = build_critic(model_cfg, obs_dim, action_dim, n_stages)
    return {
        "actor": actor,
        "critic": critic,
        "policy": actor,
        "q_network": critic,
        "config": model_cfg,
        "device": model_cfg.device,
    }


build_sac_models = build_models


def build_agent(
    cfg: Any = None,
    obs_dim: int = 0,
    action_dim: int = 0,
    n_stages: int = 1,
    device: Optional[str] = None,
    retention: Any = None,
    seed: Optional[int] = None,
    stage_id_in_obs: bool = False,
) -> Any:
    """Build a full :class:`~src.robotic_sequence.sac.SACAgent` via ``sac``."""
    build_sac_agent = _import_from_sac("build_sac_agent")
    if build_sac_agent is None:  # pragma: no cover - only without sac.py
        raise RuntimeError("src.robotic_sequence.sac.build_sac_agent is unavailable")
    return build_sac_agent(
        cfg,
        obs_dim,
        action_dim,
        n_stages=n_stages,
        device=device,
        retention=retention,
        seed=seed,
        stage_id_in_obs=stage_id_in_obs,
    )


# --------------------------------------------------------------------------------------
# Target networks / Polyak averaging (SAC tau = 0.005)
# --------------------------------------------------------------------------------------
def soft_update(target: Any, source: Any, tau: float = DEFAULT_TAU) -> None:
    """Polyak-average ``target <- tau * source + (1 - tau) * target``."""
    _require_torch()
    tau = float(tau)
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)
        for target_buffer, source_buffer in zip(target.buffers(), source.buffers()):
            target_buffer.data.copy_(source_buffer.data)


class TargetNetwork:
    """Deep-copied target network kept in sync via :func:`soft_update`."""

    def __init__(
        self,
        source: Any,
        tau: float = DEFAULT_TAU,
        device: Optional[str] = None,
    ) -> None:
        _require_torch()
        import copy as _copy

        self.tau = float(tau)
        self.source = source
        self.module = _copy.deepcopy(source)
        if device is not None:
            self.module.to(device)
        self.module.eval()
        for param in self.module.parameters():
            param.requires_grad_(False)

    def update(self) -> None:
        soft_update(self.module, self.source, self.tau)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)

    def state_dict(self) -> Dict[str, Any]:
        return self.module.state_dict()

    def load_state_dict(self, state: Mapping[str, Any], strict: bool = False) -> Any:
        return self.module.load_state_dict(state, strict=strict)

    def train(self) -> None:
        self.module.train()

    def eval(self) -> None:
        self.module.eval()

    def to(self, device: str) -> "TargetNetwork":
        self.module.to(device)
        return self


# --------------------------------------------------------------------------------------
# Optimisers (Adam, lr 1e-3, batch 128)
# --------------------------------------------------------------------------------------
def parameter_groups(model: Any, weight_decay: float = 0.0) -> List[Dict[str, Any]]:
    """Split parameters into decay / no-decay groups (biases & norms excluded)."""
    decay: List[Any] = []
    no_decay: List[Any] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lowered = name.lower()
        if param.ndim <= 1 or any(token in lowered for token in ("bias", "norm", "ln")):
            no_decay.append(param)
        else:
            decay.append(param)
    groups: List[Dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": float(weight_decay)})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    if not groups:  # param-less model (e.g. stub)
        groups.append({"params": list(model.parameters()), "weight_decay": 0.0})
    return groups


def make_optimizers(
    actor: Any,
    critic: Any,
    lr: float = DEFAULT_LR,
    weight_decay: float = 0.0,
    alpha: Optional[Any] = None,
    alpha_lr: Optional[float] = None,
) -> Dict[str, Any]:
    """Create Adam optimisers for the actor, critic and (optionally) ``alpha``."""
    _require_torch()
    optimizers: Dict[str, Any] = {
        "actor": torch.optim.Adam(
            parameter_groups(actor, weight_decay), lr=float(lr)
        ),
        "critic": torch.optim.Adam(
            parameter_groups(critic, weight_decay), lr=float(lr)
        ),
    }
    if alpha is not None:
        optimizers["alpha"] = torch.optim.Adam([alpha], lr=float(alpha_lr or lr))
    return optimizers


# --------------------------------------------------------------------------------------
# ActorCritic container
# --------------------------------------------------------------------------------------
class ActorCritic:
    """Convenience container pairing an actor with a (twin) critic.

    The SAC agent in :mod:`src.robotic_sequence.sac` owns the training loop; this
    container is used by the trainers/analysis code that only needs forward passes,
    action selection, feature extraction and checkpointing.
    """

    def __init__(
        self,
        actor: Any,
        critic: Any = None,
        config: Any = None,
        tau: float = DEFAULT_TAU,
        device: Optional[str] = None,
        target_critic: bool = False,
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.policy = actor
        self.q_network = critic
        self.config = config
        self.device = device or (config.device if config is not None else "cpu")
        self.target_critic = TargetNetwork(critic, tau=tau, device=self.device) if (
            target_critic and critic is not None
        ) else None

    # ---- forwarding -----------------------------------------------------------------
    def to(self, device: str) -> "ActorCritic":
        self.device = device
        self.actor.to(device)
        if self.critic is not None:
            self.critic.to(device)
        if self.target_critic is not None:
            self.target_critic.to(device)
        return self

    def train(self) -> "ActorCritic":
        self.actor.train()
        if self.critic is not None:
            self.critic.train()
        return self

    def eval(self) -> "ActorCritic":
        self.actor.eval()
        if self.critic is not None:
            self.critic.eval()
        if self.target_critic is not None:
            self.target_critic.eval()
        return self

    def features(self, obs: Any, stage_id: Any = None) -> Any:
        """Trunk features of the actor (used by CKA / PCA analyses)."""
        return policy_features(self.actor, obs, stage_id=stage_id)

    def distribution(self, obs: Any, stage_id: Any = None) -> Any:
        return self.actor.distribution(obs, stage_id=stage_id)

    def act(self, obs: Any, stage_id: Any = None, deterministic: bool = False, **kw: Any) -> Any:
        return self.actor.act(obs, deterministic=deterministic, stage_id=stage_id, **kw)

    def log_prob(self, obs: Any, actions: Any = None, stage_id: Any = None) -> Any:
        return self.actor.log_prob(obs, actions, stage_id=stage_id)

    def q_values(self, obs: Any, actions: Any, stage_id: Any = None) -> Any:
        if self.critic is None:
            raise RuntimeError("ActorCritic was built without a critic")
        return self.critic(obs, actions, stage_id)

    def update_targets(self) -> None:
        if self.target_critic is not None:
            self.target_critic.update()

    # ---- persistence ----------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {"actor": self.actor.state_dict()}
        if self.critic is not None:
            state["critic"] = self.critic.state_dict()
        if self.target_critic is not None:
            state["target_critic"] = self.target_critic.state_dict()
        return state

    def load_state_dict(self, state: Mapping[str, Any], strict: bool = False) -> None:
        if "actor" in state:
            self.actor.load_state_dict(state["actor"], strict=strict)
        if self.critic is not None and "critic" in state:
            self.critic.load_state_dict(state["critic"], strict=strict)
        if self.target_critic is not None and "target_critic" in state:
            self.target_critic.load_state_dict(state["target_critic"], strict=strict)

    def save(self, path: str, step: Optional[int] = None, extra: Optional[Mapping[str, Any]] = None) -> str:
        return save_model(path, self.actor, self.critic, step=step, extra=extra)

    def describe(self) -> Dict[str, Any]:
        return describe(self.actor, self.critic)


# --------------------------------------------------------------------------------------
# Analysis / retention helpers
# --------------------------------------------------------------------------------------
def policy_features(policy: Any, obs: Any, stage_id: Any = None) -> Any:
    """Trunk features of ``policy`` (pre-head), for CKA / PCA analyses."""
    _require_torch()
    for name in ("features", "encode", "trunk_features"):
        fn = getattr(policy, name, None)
        if callable(fn):
            try:
                return fn(obs, stage_id=stage_id)
            except TypeError:
                try:
                    return fn(obs)
                except Exception:
                    continue
            except Exception:
                continue
    trunk = getattr(policy, "trunk", None)
    if trunk is not None:
        return trunk(obs)
    raise RuntimeError("Could not extract features from the given policy")


def q_features(critic: Any, obs: Any, action: Any, stage_id: Any = None) -> Any:
    """Concatenated ``(obs, action)`` features of the first Q trunk."""
    _require_torch()
    x = torch.cat([obs, action], dim=-1)
    trunk = getattr(critic, "q1", None) or getattr(critic, "trunk", None)
    if trunk is None:
        raise RuntimeError("Could not locate the Q trunk")
    return trunk(x)


def layer_names(model: Any) -> List[str]:
    """Ordered names of analysable submodules (Linear/LayerNorm/activations)."""
    if model is None:
        return []
    names: List[str] = []
    for name, module in model.named_modules():
        if not name:
            continue
        if isinstance(module, (nn.Linear, nn.LayerNorm)) or module.__class__.__name__ in (
            "LeakyReLU", "ReLU", "GELU", "Tanh",
        ):
            names.append(name)
    return names


def analyzer_layers(model: Any) -> List[str]:
    """Alias used by :mod:`src.analysis.cka` when registering hooks."""
    return layer_names(model)


def actor_parameters(actor: Any, trainable_only: bool = True) -> List[Any]:
    """Flat list of actor parameters (the only ones retention losses may touch)."""
    params = list(actor.parameters())
    if trainable_only:
        params = [p for p in params if p.requires_grad]
    return params


def actor_head_parameters(actor: Any) -> List[Tuple[str, Any]]:
    """Named parameters belonging to the actor's output heads (per-stage)."""
    if _head_parameters is not None:
        for holder in (getattr(actor, "heads", None), getattr(actor, "policy_heads", None)):
            if holder is not None:
                try:
                    return list(_head_parameters(holder))
                except Exception:
                    pass
    out: List[Tuple[str, Any]] = []
    for name, param in actor.named_parameters():
        if "head" in name.lower():
            out.append((name, param))
    return out


def count_parameters(module: Any, trainable_only: bool = False) -> int:
    """Number of parameters in ``module`` (optionally only trainable ones)."""
    if module is None:
        return 0
    total = 0
    for param in module.parameters():
        if trainable_only and not param.requires_grad:
            continue
        total += int(param.numel())
    return total


def describe(actor: Any = None, critic: Any = None) -> Dict[str, Any]:
    """Compact architecture report for logging/reproducibility."""
    return {
        "actor_parameters": count_parameters(actor),
        "actor_trainable_parameters": count_parameters(actor, trainable_only=True),
        "critic_parameters": count_parameters(critic),
        "critic_trainable_parameters": count_parameters(critic, trainable_only=True),
        "actor_layers": layer_names(actor)[:16],
        "hidden_dim": HIDDEN_DIM,
        "num_hidden_layers": NUM_HIDDEN_LAYERS,
        "activation": ACTIVATION,
        "layer_norm_after_first": LAYER_NORM_AFTER_FIRST,
        "per_stage_heads": True,
        "twin_q": True,
        "optimizer": "adam",
        "lr": DEFAULT_LR,
        "batch_size": DEFAULT_BATCH_SIZE,
        "tau": DEFAULT_TAU,
        "gamma": DEFAULT_GAMMA,
    }


# --------------------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------------------
def save_model(
    path: str,
    actor: Any = None,
    critic: Any = None,
    step: Optional[int] = None,
    target_critic: Any = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> str:
    """Save actor/critic state dicts (and extras) to ``path``."""
    _require_torch()
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload: Dict[str, Any] = {}
    if actor is not None:
        payload["actor"] = actor.state_dict()
        payload["policy"] = payload["actor"]
    if critic is not None:
        payload["critic"] = critic.state_dict()
    if target_critic is not None:
        state = getattr(target_critic, "state_dict", None)
        if callable(state):
            try:
                payload["target_critic"] = state()
            except Exception:
                pass
    if step is not None:
        payload["step"] = int(step)
    if extra:
        payload.update(dict(extra))
    torch.save(payload, path)
    return path


def load_model(
    path: str,
    actor: Any = None,
    critic: Any = None,
    map_location: str = "cpu",
    strict: bool = False,
    load_actor: bool = True,
    load_critic: bool = True,
) -> Dict[str, Any]:
    """Load a checkpoint written by :func:`save_model` (or ``SACAgent.save``)."""
    _require_torch()
    payload = torch.load(path, map_location=map_location) if os.path.exists(path) else _to_state(path, map_location)
    if isinstance(payload, Mapping) and "state_dict" in payload and len(payload) == 1:
        payload = payload["state_dict"]

    if actor is not None and load_actor:
        state = None
        if isinstance(payload, Mapping):
            state = payload.get("actor", payload.get("policy"))
            if state is None and all(isinstance(k, str) for k in payload.keys()):
                state = payload  # raw actor state dict
        if state is not None:
            actor.load_state_dict(state, strict=strict)
    if critic is not None and load_critic and isinstance(payload, Mapping):
        state = payload.get("critic")
        if state is not None:
            critic.load_state_dict(state, strict=strict)
    return payload if isinstance(payload, Mapping) else {"state_dict": payload}


def _to_state(payload: Any, map_location: str = "cpu") -> Any:
    """Accept either a path or an in-memory state dict."""
    if isinstance(payload, str) and not os.path.exists(payload):
        return {}
    if isinstance(payload, Mapping):
        return payload
    if _HAS_TORCH:
        try:
            return torch.load(payload, map_location=map_location)
        except Exception:
            return {}
    return {}


# --------------------------------------------------------------------------------------
# CLI smoke test
# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RoboticSequence SAC model smoke test / architecture report"
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--set", dest="overrides", nargs="*", default=None)
    parser.add_argument("--obs-dim", type=int, default=9)
    parser.add_argument("--action-dim", type=int, default=4)
    parser.add_argument("--n-stages", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-smoke-test", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    import json

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg: Any = None
    if args.config:
        load_config = None
        for module_name in ("src.common.config", "common.config"):
            try:
                load_config = __import__(module_name, fromlist=["load_config"]).load_config
                break
            except Exception:
                continue
        if load_config is not None:
            cfg = load_config(args.config)

    overrides = {}
    if args.set:
        for item in args.set:
            if "=" in item:
                key, value = item.split("=", 1)
                overrides[key.strip()] = value.strip()

    if not _HAS_TORCH:
        print(json.dumps({"torch": False, "message": "PyTorch unavailable"}, indent=2))
        return 0

    models = build_models(
        cfg, args.obs_dim, args.action_dim, args.n_stages, device=args.device, **overrides
    )
    report = describe(models["actor"], models["critic"])
    report["device"] = args.device
    report["obs_dim"] = args.obs_dim
    report["action_dim"] = args.action_dim
    report["n_stages"] = args.n_stages

    if not args.no_smoke_test:
        obs = torch.randn(args.batch_size, args.obs_dim)
        stage_ids = torch.randint(0, args.n_stages, (args.batch_size,))
        dist = models["actor"].distribution(obs, stage_ids)
        actions = dist.sample()
        log_prob = dist.log_prob(actions)
        q1, q2 = models["critic"](obs, actions, stage_ids)
        report["smoke_test"] = {
            "action_shape": list(actions.shape),
            "log_prob_shape": list(log_prob.shape),
            "q_values_shape": [list(q1.shape), list(q2.shape)],
            "finite": bool(
                torch.isfinite(actions).all()
                and torch.isfinite(log_prob).all()
                and torch.isfinite(q1).all()
                and torch.isfinite(q2).all()
            ),
        }
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
