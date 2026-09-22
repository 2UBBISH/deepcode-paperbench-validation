"""Policy and mask-network architectures for RICE.

This module is the single source of truth for the **model layer** of the RICE
pipeline (paper: *RICE: A Refining Scheme for Reinforcement Learning with
Explanation*, ICML 2024, PMLR 235).

Paper / addendum specification (Appendix C.1 "Implementation Details" +
addendum section "Architectures"):

> We utilized the default setting ('MlpPolicy') in stable baselines 3 library for
> the policy network and SAC networks in dense/sparse MuJoCo tasks.
> We utilized an MLP for the policy network in the selfish mining task. The
> hidden sizes are [128, 128, 128, 128].
> We utilized an MLP for the policy network in the cage challenge task. The
> hidden sizes are [64, 64, 64].
> We utilized the default network structure (DI-engine VAC) for the policy
> network in the autonomous driving task.

The mask network (Stage 1 / explanation, Algorithm 1) has the *same* body
architecture as the target policy and a binary head (two logits over
``{keep=0, blind=1}``); the state-importance score is the probability that the
mask outputs ``0`` ("keep") -- see Sec. 3.3 ("Step-level Explanation") and
Algorithm 1.

Everything here is defensive: PyTorch is a hard requirement for the model
layer, but Stable-Baselines3 is optional (we fall back to the native
:class:`ActorCritic` implementation when SB3 is unavailable).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "ARCHS",
    "DEFAULT_HIDDEN_SIZES",
    "MASK_ARCHS",
    "POLICY_ARCHS",
    "ActorCritic",
    "MaskArch",
    "MLP",
    "build_mlp",
    "build_policy",
    "describe_arch",
    "load_policy",
    "mask_arch",
    "normalize_env_key",
    "policy_arch",
    "sample_random_action",
    "save_policy",
    "sb3_net_arch",
    "sb3_policy_kwargs",
]

# ---------------------------------------------------------------------------
# Optional heavy dependencies
# ---------------------------------------------------------------------------
try:  # pragma: no cover - environment dependent
    import torch
    import torch.nn as nn

    PYTORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    PYTORCH_AVAILABLE = False

try:  # pragma: no cover - environment dependent
    import stable_baselines3  # noqa: F401
    from stable_baselines3.common.policies import ActorCriticPolicy as _SB3ActorCriticPolicy

    SB3_AVAILABLE = True
except Exception:  # pragma: no cover
    _SB3ActorCriticPolicy = None  # type: ignore
    SB3_AVAILABLE = False


# ---------------------------------------------------------------------------
# Architecture table
# ---------------------------------------------------------------------------
# SB3's default ``MlpPolicy`` uses a two-layer tanh MLP of width 64 for both the
# policy ("pi") and value ("vf") networks -- see
# https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
DEFAULT_HIDDEN_SIZES: Tuple[int, ...] = (64, 64)
DEFAULT_ACTIVATION: str = "tanh"

# Selfish Mining / CAGE-2 hidden sizes are given explicitly by the authors.
SELFISH_MINING_HIDDEN: Tuple[int, ...] = (128, 128, 128, 128)
CAGE2_HIDDEN: Tuple[int, ...] = (64, 64, 64)

# DI-engine's default VAC template (encoder MLP + policy/value heads).
# See https://di-engine-docs.readthedocs.io/en/latest/_modules/ding/model/template/vac.html
VAC_DEFAULT_HIDDEN: Tuple[int, ...] = (64, 64)


def _arch(
    policy_hidden: Sequence[int],
    mask_hidden: Optional[Sequence[int]] = None,
    activation: str = DEFAULT_ACTIVATION,
    backend: str = "sb3",
    sb3_default: bool = False,
    obs_dim: Optional[int] = None,
    action_dim: Optional[int] = None,
    discrete: bool = False,
    note: str = "",
) -> Dict[str, Any]:
    """Small helper building one entry of the :data:`ARCHS` table."""
    return {
        "policy_hidden": tuple(int(h) for h in policy_hidden),
        "mask_hidden": tuple(int(h) for h in (mask_hidden if mask_hidden is not None else policy_hidden)),
        "activation": str(activation),
        "backend": str(backend),
        "sb3_default": bool(sb3_default),
        "obs_dim": None if obs_dim is None else int(obs_dim),
        "action_dim": None if action_dim is None else int(action_dim),
        "discrete": bool(discrete),
        "note": str(note),
    }


#: Per-application architecture table (keys are canonical env keys produced by
#: :func:`normalize_env_key`).
ARCHS: Dict[str, Dict[str, Any]] = {
    # ---- dense MuJoCo (SB3 default MlpPolicy) -----------------------------
    "hopper": _arch(
        DEFAULT_HIDDEN_SIZES,
        backend="sb3",
        sb3_default=True,
        obs_dim=11,
        action_dim=3,
        note="Stable-Baselines3 default MlpPolicy (Tanh MLP 64x2 for pi and vf).",
    ),
    "walker2d": _arch(
        DEFAULT_HIDDEN_SIZES,
        backend="sb3",
        sb3_default=True,
        obs_dim=17,
        action_dim=6,
        note="Stable-Baselines3 default MlpPolicy; observations normalized (App. C.2).",
    ),
    "reacher": _arch(
        DEFAULT_HIDDEN_SIZES,
        backend="sb3",
        sb3_default=True,
        obs_dim=11,
        action_dim=2,
        note="Stable-Baselines3 default MlpPolicy (Reacher-v2).",
    ),
    "halfcheetah": _arch(
        DEFAULT_HIDDEN_SIZES,
        backend="sb3",
        sb3_default=True,
        obs_dim=17,
        action_dim=6,
        note="Stable-Baselines3 default MlpPolicy; observations normalized (App. C.2).",
    ),
    # ---- sparse MuJoCo (SB3 default MlpPolicy) ---------------------------
    "sparse_hopper": _arch(
        DEFAULT_HIDDEN_SIZES,
        backend="sb3",
        sb3_default=True,
        obs_dim=11,
        action_dim=3,
        note="Sparse reward variant of Hopper (App. C.2); SB3 default MlpPolicy.",
    ),
    "sparse_halfcheetah": _arch(
        DEFAULT_HIDDEN_SIZES,
        backend="sb3",
        sb3_default=True,
        obs_dim=17,
        action_dim=6,
        note="Sparse reward variant of HalfCheetah (App. C.2); SB3 default MlpPolicy.",
    ),
    # ---- blockchain: selfish mining --------------------------------------
    "selfish_mining": _arch(
        SELFISH_MINING_HIDDEN,
        backend="native",
        obs_dim=None,
        action_dim=3,
        discrete=True,
        note="MLP with hidden sizes [128, 128, 128, 128]; 3 actions Adopt/Reveal/Mine.",
    ),
    # ---- cyber: CAGE Challenge 2 -----------------------------------------
    "cage2": _arch(
        CAGE2_HIDDEN,
        backend="native",
        obs_dim=None,
        action_dim=None,
        discrete=True,
        note="MLP with hidden sizes [64, 64, 64] (CAGE Challenge 2 blue agent).",
    ),
    # ---- autonomous driving: MetaDrive Macro-v1 --------------------------
    "autodriving": _arch(
        VAC_DEFAULT_HIDDEN,
        backend="di-engine",
        obs_dim=None,
        action_dim=2,
        note="DI-engine default VAC template (see addendum 'Architectures').",
    ),
    # ---- fallback --------------------------------------------------------
    "default": _arch(
        DEFAULT_HIDDEN_SIZES,
        backend="sb3",
        sb3_default=True,
        note="Fallback architecture (SB3 default MlpPolicy).",
    ),
}

#: Convenience views: canonical env key -> hidden sizes.
POLICY_ARCHS: Dict[str, Tuple[int, ...]] = {k: v["policy_hidden"] for k, v in ARCHS.items()}
MASK_ARCHS: Dict[str, Tuple[int, ...]] = {k: v["mask_hidden"] for k, v in ARCHS.items()}


# ---------------------------------------------------------------------------
# Environment-key normalisation
# ---------------------------------------------------------------------------
_ALIASES: Dict[str, str] = {
    # MuJoCo dense
    "hopper": "hopper",
    "hoper": "hopper",
    "sparsehopper": "sparse_hopper",
    "sparse_hopper": "sparse_hopper",
    "walker": "walker2d",
    "walker2d": "walker2d",
    "walker_2d": "walker2d",
    "walker2ddsl": "walker2d",
    "reacher": "reacher",
    "halfcheetah": "halfcheetah",
    "half_cheetah": "halfcheetah",
    "sparsehalfcheetah": "sparse_halfcheetah",
    "sparse_halfcheetah": "sparse_halfcheetah",
    # applications
    "selfishmining": "selfish_mining",
    "selfish_mining": "selfish_mining",
    "selfish": "selfish_mining",
    "cage2": "cage2",
    "cage_2": "cage2",
    "cagechallenge2": "cage2",
    "cagechallenge_2": "cage2",
    "cage": "cage2",
    "autodriving": "autodriving",
    "autonomousdriving": "autodriving",
    "autonomous_driving": "autodriving",
    "metadrive": "autodriving",
    "metadrivemacro": "autodriving",
    "metadrive_macro": "autodriving",
    "driving": "autodriving",
    # out of scope but recognised
    "malwaremutation": "malware_mutation",
    "malware_mutation": "malware_mutation",
    "malconv": "malware_mutation",
    "default": "default",
}


def normalize_env_key(env_id: Any) -> str:
    """Canonicalise an environment identifier to a key of :data:`ARCHS`.

    Handles gym ids (``Hopper-v3``), RICE ids (``SparseHopper-v0``,
    ``SelfishMining-v0``, ``Cage2-v0``, ``MetaDrive-Macro-v1``), config/stem
    names (``selfish_mining``, ``cage2``, ``autodriving``) and free-form labels
    such as ``CAGE Challenge 2``.
    """
    if env_id is None:
        return "default"
    key = str(env_id).strip().lower()
    key = key.split("/")[-1].split("\\")[-1]
    key = re.sub(r"\.(yaml|yml|json|py)$", "", key)
    key = re.sub(r"[-_]?v\d+$", "", key)          # drop version suffix (-v3/_v0)
    key = re.sub(r"[-_\s:]+", "_", key).strip("_")
    if key in _ALIASES:
        return _ALIASES[key]
    compact = key.replace("_", "")
    if compact in _ALIASES:
        return _ALIASES[compact]
    # prefix heuristics (e.g. "sparsewalker2d")
    if compact.startswith("sparse") and compact[6:] in _ALIASES:
        base = _ALIASES[compact[6:]]
        return base if base.startswith("sparse") else "sparse_" + base
    for alias, canonical in _ALIASES.items():
        if alias and alias in compact and canonical != "default":
            return canonical
    return "default"


def _spec(env_id: Any) -> Dict[str, Any]:
    """Return the architecture entry (never ``None``) for ``env_id``."""
    return ARCHS[normalize_env_key(env_id)]


def policy_arch(env_id: Any) -> Tuple[int, ...]:
    """Hidden-layer widths of the target policy pi for ``env_id``."""
    hidden = _spec(env_id)["policy_hidden"]
    return tuple(int(h) for h in hidden)


def describe_arch(env_id: Any = "default", kind: str = "policy") -> str:
    """Human-readable architecture description (used for logging/reporting)."""
    key = normalize_env_key(env_id)
    spec = ARCHS[key]
    hidden = spec["mask_hidden"] if str(kind).lower().startswith("mask") else spec["policy_hidden"]
    extra = []
    if spec["sb3_default"]:
        extra.append("SB3 default MlpPolicy")
    if spec["discrete"]:
        extra.append("discrete head")
    if spec["action_dim"] is not None:
        extra.append("action_dim=%d" % spec["action_dim"])
    if spec["obs_dim"] is not None:
        extra.append("obs_dim=%d" % spec["obs_dim"])
    suffix = ("; " + ", ".join(extra)) if extra else ""
    return "%s (%s): hidden=%s activation=%s backend=%s%s" % (
        key,
        kind,
        list(hidden),
        spec["activation"],
        spec["backend"],
        suffix,
    )


# ---------------------------------------------------------------------------
# Network building blocks
# ---------------------------------------------------------------------------
def get_activation(name: Any = "tanh"):
    """Resolve an activation callable/module factory from a name or callable."""
    if callable(name) and not isinstance(name, str):
        return name
    if not PYTORCH_AVAILABLE:  # pragma: no cover
        raise RuntimeError("PyTorch is required to build networks.")
    table = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "identity": nn.Identity,
        "linear": nn.Identity,
        "none": nn.Identity,
    }
    key = str(name).strip().lower()
    if key not in table:
        raise ValueError("Unknown activation '%s' (choices: %s)." % (name, sorted(table)))
    return table[key]


def orthogonal_init(module, gain: float = np.sqrt(2.0)) -> None:
    """Orthogonal initialisation of Linear layers (SB3-style)."""
    if not PYTORCH_AVAILABLE:  # pragma: no cover
        return
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            module.bias.data.fill_(0.0)


def build_mlp(
    input_dim: int,
    hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
    output_dim: Optional[int] = None,
    activation: Any = DEFAULT_ACTIVATION,
    output_activation: Any = None,
    layer_norm: bool = False,
    init_orthogonal: bool = True,
    squash_output: bool = False,
    **_: Any,
):
    """Build a fully connected MLP.

    Parameters
    ----------
    input_dim, output_dim:
        Input/output feature widths.  ``output_dim=None`` yields a body-only MLP
        (final layer is the last hidden layer).
    hidden_sizes:
        Widths of hidden layers, e.g. ``(64, 64)`` (SB3 default) or
        ``(128, 128, 128, 128)`` (Selfish Mining) / ``(64, 64, 64)`` (CAGE-2).
    activation, output_activation:
        Activation names (see :func:`get_activation`) or callables.
    """
    if not PYTORCH_AVAILABLE:  # pragma: no cover
        raise RuntimeError("PyTorch is required to build an MLP (pip install torch).")

    hidden_sizes = [int(h) for h in (hidden_sizes or [])]
    layers: List[nn.Module] = []
    last = int(input_dim)
    for width in hidden_sizes:
        layers.append(nn.Linear(last, width))
        if layer_norm:
            layers.append(nn.LayerNorm(width))
        layers.append(get_activation(activation)())
        last = width
    if output_dim is not None:
        layers.append(nn.Linear(last, int(output_dim)))
        if output_activation is not None:
            layers.append(get_activation(output_activation)())
        else:
            if init_orthogonal:
                # initialise the output layer with a small gain
                try:
                    nn.init.orthogonal_(layers[-1].weight, gain=0.01)
                    nn.init.constant_(layers[-1].bias, 0.0)
                except Exception:  # pragma: no cover
                    pass
    seq = nn.Sequential(*layers)
    if init_orthogonal:
        for m in seq.modules():
            if isinstance(m, nn.Linear):
                orthogonal_init(m)
        if output_dim is not None:
            # redo the output layer with the small gain chosen above
            try:
                nn.init.orthogonal_(layers[-1].weight, gain=0.01)
                nn.init.constant_(layers[-1].bias, 0.0)
            except Exception:  # pragma: no cover
                pass
    if squash_output:
        seq.add_module("squash", nn.Tanh())
    return seq


if PYTORCH_AVAILABLE:

    class MLP(nn.Module):
        """A plain multi-layer perceptron (body or body+head)."""

        def __init__(
            self,
            input_dim: int,
            hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
            output_dim: Optional[int] = None,
            activation: Any = DEFAULT_ACTIVATION,
            output_activation: Any = None,
            layer_norm: bool = False,
            init_orthogonal: bool = True,
            squash_output: bool = False,
        ) -> None:
            super().__init__()
            self.input_dim = int(input_dim)
            self.hidden_sizes = tuple(int(h) for h in (hidden_sizes or []))
            self.output_dim = None if output_dim is None else int(output_dim)
            self.activation = activation
            self.net = build_mlp(
                self.input_dim,
                self.hidden_sizes,
                self.output_dim,
                activation=activation,
                output_activation=output_activation,
                layer_norm=layer_norm,
                init_orthogonal=init_orthogonal,
                squash_output=squash_output,
            )

        @property
        def output_size(self) -> int:
            return self.output_dim if self.output_dim is not None else (
                self.hidden_sizes[-1] if self.hidden_sizes else self.input_dim
            )

        def forward(self, obs):
            return self.net(obs)

        def extra_repr(self) -> str:  # pragma: no cover - cosmetic
            return "input=%d hidden=%s output=%s" % (
                self.input_dim,
                list(self.hidden_sizes),
                self.output_dim,
            )


    class ActorCritic(nn.Module):
        """Native actor-critic used when SB3 is not available.

        Supports both continuous (diagonal Gaussian with tanh squashing into the
        action-space bounds) and discrete (categorical) action spaces.  The
        Stage-1 mask network reuses this module as a 2-way discrete actor
        (``action_dim=2`` → two logits over ``{keep=0, blind=1}``).
        """

        def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
            activation: Any = DEFAULT_ACTIVATION,
            discrete: bool = False,
            log_std_init: float = -1.0,
            shared_body: bool = False,
            squash: bool = True,
            action_low: Optional[Sequence[float]] = None,
            action_high: Optional[Sequence[float]] = None,
            **_: Any,
        ) -> None:
            super().__init__()
            self.obs_dim = int(obs_dim)
            self.action_dim = int(action_dim)
            self.discrete = bool(discrete)
            self.squash = bool(squash)
            self.hidden_sizes = tuple(int(h) for h in (hidden_sizes or []))
            self.activation = activation

            self.shared_body = None
            if shared_body:
                self.shared_body = build_mlp(self.obs_dim, self.hidden_sizes, None, activation=activation)
                body_out = self.shared_body.output_size  # type: ignore[attr-defined]
                pi_in = vf_in = body_out
            else:
                pi_in = vf_in = self.obs_dim

            self.pi = build_mlp(pi_in, self.hidden_sizes, self.action_dim, activation=activation)
            self.vf = build_mlp(vf_in, self.hidden_sizes, 1, activation=activation)
            # value head: standard gain-1 output initialisation
            try:
                nn.init.orthogonal_(self.vf[-1].weight, gain=1.0)
                nn.init.constant_(self.vf[-1].bias, 0.0)
            except Exception:  # pragma: no cover
                pass

            if not self.discrete:
                self.log_std = nn.Parameter(torch.ones(self.action_dim) * float(log_std_init))
            if action_low is not None and action_high is not None:
                self.set_action_space(action_low, action_high)

        # -- action space ---------------------------------------------------
        def set_action_space(self, low: Sequence[float], high: Sequence[float]) -> "ActorCritic":
            low = np.asarray(low, dtype=np.float32).reshape(-1)
            high = np.asarray(high, dtype=np.float32).reshape(-1)
            if low.size == 1:  # scalar bounds broadcast
                low = np.repeat(low, self.action_dim)
            if high.size == 1:
                high = np.repeat(high, self.action_dim)
            self.register_buffer("action_low", torch.as_tensor(low, dtype=torch.float32))
            self.register_buffer("action_high", torch.as_tensor(high, dtype=torch.float32))
            return self

        def _body(self, obs):
            if self.shared_body is not None:
                return self.shared_body(obs)
            return obs

        # -- distributions --------------------------------------------------
        def get_distribution(self, obs):
            h = self._body(obs)
            if self.discrete:
                return torch.distributions.Categorical(logits=self.pi(h))
            mean = self.pi(h)
            std = torch.exp(self.log_std).expand_as(mean)
            return torch.distributions.Normal(mean, std)

        def predict_values(self, obs):
            return self.vf(self._body(obs)).squeeze(-1)

        def forward(self, obs):
            """Return ``(mean_or_logits, value)``."""
            h = self._body(obs)
            return self.pi(h), self.vf(h).squeeze(-1)

        def _squash(self, raw, deterministic: bool = False):
            if self.discrete or not self.squash:
                return raw
            if deterministic:
                squashed = torch.tanh(raw)
            else:
                squashed = torch.tanh(raw)
            if hasattr(self, "action_low") and hasattr(self, "action_high"):
                low = self.action_low.to(squashed.device)
                high = self.action_high.to(squashed.device)
                return low + (0.5 * (squashed + 1.0)) * (high - low)
            return squashed

        @staticmethod
        def _as_tensor(obs):
            if torch.is_tensor(obs):
                return obs.float()
            return torch.as_tensor(np.asarray(obs), dtype=torch.float32)

        def evaluate_actions(self, obs, actions):
            obs_t = self._as_tensor(obs)
            dist = self.get_distribution(obs_t)
            values = self.predict_values(obs_t)
            if self.discrete:
                act_t = self._as_tensor(actions).long().reshape(-1)
                log_prob = dist.log_prob(act_t)
            else:
                act_t = self._as_tensor(actions)
                if act_t.dim() == 1:
                    act_t = act_t.unsqueeze(0)
                log_prob = dist.log_prob(act_t).sum(dim=-1)
            entropy = dist.entropy()
            if entropy.dim() > 1:
                entropy = entropy.sum(dim=-1)
            return log_prob, entropy, values

        # -- acting ---------------------------------------------------------
        def act(self, obs, deterministic: bool = False):
            """Return ``(action, value, log_prob)`` for a single observation."""
            obs_t = self._as_tensor(obs)
            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0)
            dist = self.get_distribution(obs_t)
            if self.discrete:
                if deterministic:
                    action = torch.argmax(dist.logits, dim=-1)
                else:
                    action = dist.sample()
            else:
                raw = dist.mean if deterministic else dist.sample()
                action = self._squash(raw, deterministic=deterministic)
            log_prob = dist.log_prob(dist.mean if self.discrete else dist.loc if deterministic else
                                     (torch.nn.functional.one_hot(action.long(), self.action_dim).float()
                                      if self.discrete else action))
            value = self.predict_values(obs_t)
            if self.discrete:
                return int(action.item()), float(value.item()), log_prob
            return action.detach().cpu().numpy().reshape(-1), float(value.item()), log_prob

        @torch.no_grad()
        def predict(self, obs, deterministic: bool = False):
            """SB3-compatible ``predict`` returning ``(action, None)``."""
            action, _, _ = self.act(obs, deterministic=deterministic)
            return action, None

        def extra_repr(self) -> str:  # pragma: no cover - cosmetic
            return "obs=%d action=%d hidden=%s discrete=%s" % (
                self.obs_dim,
                self.action_dim,
                list(self.hidden_sizes),
                self.discrete,
            )

else:  # pragma: no cover - only when torch is missing

    class MLP:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("PyTorch is required for rice.models.policies (pip install torch).")

    class ActorCritic(MLP):  # type: ignore
        pass


# ---------------------------------------------------------------------------
# Mask-network architecture specification
# ---------------------------------------------------------------------------
@dataclass
class MaskArch:
    """Specification of the Stage-1 mask network :math:`\\tilde{\\pi}_\\theta`.

    The mask takes the state ``s_t`` as input and outputs a binary action
    ``a_t^m in {0, 1}`` (two logits / Bernoulli).  ``a_t^m = 0`` means "keep the
    target action" and ``a_t^m = 1`` means "blind the target agent" (replace the
    action by a random one).  The **state importance** is
    ``P(a_t^m = 0 | s_t)`` (Sec. 3.3, Algorithm 1).
    """

    env_key: str = "default"
    obs_dim: Optional[int] = None
    hidden_sizes: Tuple[int, ...] = DEFAULT_HIDDEN_SIZES
    activation: str = DEFAULT_ACTIVATION
    num_logits: int = 2
    backend: str = "native"
    keep_index: int = 0
    blind_index: int = 1

    def to_sb3_net_arch(self) -> Dict[str, List[int]]:
        return {"pi": list(self.hidden_sizes), "vf": list(self.hidden_sizes)}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def output_dim(self) -> int:
        """Number of output logits (2: keep / blind)."""
        return int(self.num_logits)


def mask_arch(env_id: Any = "default", obs_dim: Optional[int] = None, **kwargs: Any) -> MaskArch:
    """Resolve the mask-network architecture for ``env_id``.

    The mask body mirrors the target policy's body (addendum "Architectures")
    and its head is a 2-way discrete head.
    """
    key = normalize_env_key(env_id)
    spec = ARCHS[key]
    return MaskArch(
        env_key=key,
        obs_dim=None if obs_dim is None else int(obs_dim),
        hidden_sizes=tuple(spec["mask_hidden"]),
        activation=str(spec["activation"]),
        num_logits=int(kwargs.pop("num_logits", 2)),
        backend=str(spec["backend"]),
    )


# ---------------------------------------------------------------------------
# Stable-Baselines3 glue
# ---------------------------------------------------------------------------
def sb3_net_arch(env_id_or_hidden: Any = None, **kwargs: Any) -> Dict[str, List[int]]:
    """Build the SB3 ``net_arch`` dict (``{"pi": [...], "vf": [...]}``)."""
    kind = str(kwargs.pop("kind", "policy")).lower()
    hidden = _resolve_hidden(env_id_or_hidden, kind=kind)
    return {"pi": list(hidden), "vf": list(hidden)}


def sb3_policy_kwargs(
    env_id_or_hidden: Any = None,
    activation: Any = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Build a ``policy_kwargs`` dict for SB3 ``PPO``/``SAC`` (defaults: SB3 MLP)."""
    kind = str(kwargs.pop("kind", "policy")).lower()
    hidden = _resolve_hidden(env_id_or_hidden, kind=kind)
    if activation is None:
        activation = ARCHS[normalize_env_key(kwargs.pop("env_id", env_id_or_hidden))]["activation"]
    policy_kwargs: Dict[str, Any] = {"net_arch": {"pi": list(hidden), "vf": list(hidden)}}
    if PYTORCH_AVAILABLE:
        try:
            policy_kwargs["activation_fn"] = get_activation(activation)
        except Exception:  # pragma: no cover
            pass
    if extra:
        policy_kwargs.update(dict(extra))
    return policy_kwargs


def _resolve_hidden(env_id_or_hidden: Any = None, kind: str = "policy") -> Tuple[int, ...]:
    """Accept hidden sizes, an env id, or ``None`` and return hidden sizes."""
    if env_id_or_hidden is None:
        return DEFAULT_HIDDEN_SIZES
    if isinstance(env_id_or_hidden, (list, tuple, np.ndarray)):
        seq = [int(h) for h in env_id_or_hidden]
        return tuple(seq) if seq else DEFAULT_HIDDEN_SIZES
    if isinstance(env_id_or_hidden, int):
        return (int(env_id_or_hidden),)
    spec = ARCHS[normalize_env_key(env_id_or_hidden)]
    return tuple(spec["mask_hidden"] if str(kind).lower().startswith("mask") else spec["policy_hidden"])


# ---------------------------------------------------------------------------
# Policy factory
# ---------------------------------------------------------------------------
def _obs_dim_from_space(space: Any) -> Optional[int]:
    if space is None:
        return None
    shape = getattr(space, "shape", None)
    if shape is None:
        return None
    n = int(np.prod(shape))
    return n


def _normalise_action_space(action_space: Any, action_dim: Optional[int]) -> Tuple[bool, int, Any, Any]:
    """Return ``(discrete, action_dim, low, high)`` for the given space."""
    if action_space is None:
        d = int(action_dim if action_dim is not None else 1)
        return False, d, None, None
    if isinstance(action_space, (int, np.integer)):
        return False, int(action_space), None, None
    name = type(action_space).__name__.lower()
    if "discrete" in name or hasattr(action_space, "n"):
        n = int(getattr(action_space, "n", action_dim or 2))
        return True, n, None, None
    low = getattr(action_space, "low", None)
    high = getattr(action_space, "high", None)
    d = int(np.prod(getattr(action_space, "shape", (action_dim or 1,))))
    return False, d, low, high


def build_policy(
    env_id: Any = "default",
    obs_dim: Optional[int] = None,
    action_dim: Optional[int] = None,
    action_space: Any = None,
    observation_space: Any = None,
    kind: str = "policy",
    backend: str = "auto",
    hidden_sizes: Optional[Sequence[int]] = None,
    activation: Any = None,
    discrete: Optional[bool] = None,
    device: Any = "cpu",
    **kwargs: Any,
):
    """Build a policy / mask-network module.

    ``kind="policy"`` returns the target policy pi; ``kind="mask"`` returns a
    2-way discrete actor (two logits over keep/blind) used by the Stage-1 mask
    network.

    ``backend``:

    * ``"native"`` – always return the native :class:`ActorCritic`;
    * ``"sb3"`` – return an SB3 ``ActorCriticPolicy`` when available (requires
      gym spaces); falls back to native otherwise;
    * ``"auto"`` – SB3 for MuJoCo-style keys when spaces/sizes are available,
      native otherwise.
    """
    key = normalize_env_key(env_id)
    spec = ARCHS[key]
    is_mask = str(kind).lower().startswith("mask")

    if activation is None:
        activation = spec["activation"]
    if hidden_sizes is None:
        hidden_sizes = spec["mask_hidden"] if is_mask else spec["policy_hidden"]
    hidden_sizes = tuple(int(h) for h in hidden_sizes)

    if obs_dim is None:
        obs_dim = _obs_dim_from_space(observation_space)
    if obs_dim is None:
        obs_dim = spec["obs_dim"]

    if is_mask:
        # the mask is a Bernoulli / 2-logit head over {keep, blind}
        act_dim = int(kwargs.pop("mask_logits", 2))
        is_discrete = True
    else:
        is_discrete, act_dim, low, high = _normalise_action_space(action_space, action_dim)
        if discrete is not None:
            is_discrete = bool(discrete)
        low = kwargs.pop("action_low", low)
        high = kwargs.pop("action_high", high)

    backend = str(backend).lower()
    if backend == "auto":
        backend = "sb3" if (SB3_AVAILABLE and not is_mask and spec["backend"] == "sb3"
                            and observation_space is not None and action_space is not None) else "native"

    if backend == "sb3" and SB3_AVAILABLE and not is_mask:
        try:
            return _build_sb3_policy(
                observation_space,
                action_space,
                hidden_sizes=hidden_sizes,
                activation=activation,
                device=device,
                **kwargs,
            )
        except Exception:
            # fall through to the native implementation
            pass

    if obs_dim is None:
        raise ValueError(
            "build_policy(%r): obs_dim could not be inferred; pass obs_dim or "
            "observation_space." % (env_id,)
        )
    if act_dim is None:
        raise ValueError(
            "build_policy(%r): action_dim could not be inferred; pass action_dim or "
            "action_space." % (env_id,)
        )

    model = ActorCritic(
        obs_dim=int(obs_dim),
        action_dim=int(act_dim),
        hidden_sizes=hidden_sizes,
        activation=activation,
        discrete=bool(is_discrete),
        **kwargs,
    )
    if not is_discrete and (low is not None and high is not None):
        try:
            model.set_action_space(low, high)
        except Exception:  # pragma: no cover
            pass
    try:
        model = model.to(device)
    except Exception:  # pragma: no cover
        pass
    return model


def _build_sb3_policy(
    observation_space: Any,
    action_space: Any,
    hidden_sizes: Sequence[int],
    activation: Any,
    device: Any = "cpu",
    learning_rate: float = 3e-4,
    **kwargs: Any,
):
    """Instantiate an SB3 ``ActorCriticPolicy`` (default MlpPolicy) directly."""
    if not SB3_AVAILABLE:  # pragma: no cover
        raise RuntimeError("stable_baselines3 is not installed")

    def lr_schedule(_progress: float) -> float:
        return float(learning_rate)

    policy_kwargs = {"net_arch": {"pi": list(hidden_sizes), "vf": list(hidden_sizes)}}
    try:
        policy_kwargs["activation_fn"] = get_activation(activation)
    except Exception:  # pragma: no cover
        pass
    for extra_key in ("ortho_init", "log_std_init", "squash_output", "features_extractor_kwargs"):
        if extra_key in kwargs:
            policy_kwargs[extra_key] = kwargs.pop(extra_key)
    return _SB3ActorCriticPolicy(
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=lr_schedule,
        net_arch=policy_kwargs["net_arch"],
        activation_fn=policy_kwargs.get("activation_fn", None),
        **{k: v for k, v in policy_kwargs.items() if k not in ("net_arch", "activation_fn")},
        device=device,
    )


# ---------------------------------------------------------------------------
# Random-action sampling (masked-action operator helper)
# ---------------------------------------------------------------------------
def sample_random_action(
    action_space: Any = None,
    rng: Any = None,
    dim: Optional[int] = None,
    discrete: bool = False,
    low: Any = None,
    high: Any = None,
    dtype: Any = np.float32,
):
    """Sample a uniformly random action.

    Used by the masked-action operator ``a = a_t`` if ``a_t^m = 0`` else
    ``a_random`` (Sec. 3.3).  Accepts a gym space, an integer action-dim, or
    explicit ``dim``/``low``/``high``.
    """
    rng = np.random if rng is None else rng
    uniform = getattr(rng, "uniform", None)
    randint = getattr(rng, "randint", None)
    if uniform is None:  # a torch Generator or similar
        rng = np.random
        uniform = rng.uniform
        randint = rng.randint

    if action_space is not None and not isinstance(action_space, (int, np.integer)):
        name = type(action_space).__name__.lower()
        if "discrete" in name or hasattr(action_space, "n"):
            n = int(getattr(action_space, "n", 2))
            if randint is not None:
                return int(randint(n))
            return int(np.random.randint(n))
        if "multi" in name and "binary" in name:
            n = int(np.prod(getattr(action_space, "shape", (dim or 1,))))
            return (rng.random(n) < 0.5).astype(np.int8) if hasattr(rng, "random") else np.random.randint(2, size=n)
        low = getattr(action_space, "low", low)
        high = getattr(action_space, "high", high)
        if dim is None:
            shape = getattr(action_space, "shape", None)
            dim = int(np.prod(shape)) if shape is not None else 1
        try:  # gym Box: honour declared bounds (may be +-inf)
            sample = action_space.sample()
            return np.asarray(sample, dtype=np.float32).reshape(-1)
        except Exception:
            pass

    if discrete:
        n = int(dim if dim is not None else 2)
        return int(rng.randint(n) if randint is not None else np.random.randint(n))

    d = int(dim if dim is not None else 1)
    if low is None:
        low = np.full(d, -1.0, dtype=np.float32)
    if high is None:
        high = np.full(d, 1.0, dtype=np.float32)
    low_arr = np.asarray(low, dtype=np.float32).reshape(-1)
    high_arr = np.asarray(high, dtype=np.float32).reshape(-1)
    if low_arr.size == 1:
        low_arr = np.repeat(low_arr, d)
    if high_arr.size == 1:
        high_arr = np.repeat(high_arr, d)
    finite = np.isfinite(low_arr) & np.isfinite(high_arr)
    out = np.zeros(d, dtype=dtype)
    if finite.any():
        out[finite] = uniform(low_arr[finite], high_arr[finite])
    # unbounded coordinates: standard-normal-like scale, matching Box(-inf, inf) sampling
    if (~finite).any():
        unbounded = np.full(int((~finite).sum()), 1.0, dtype=np.float32)
        normal = getattr(rng, "normal", None) or getattr(np.random, "normal")
        out[~finite] = normal(0.0, 1.0, size=unbounded.size)
    return out.astype(dtype)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def _checkpoint_payload(policy: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"format": "rice-policy", "version": 1}
    if hasattr(policy, "state_dict"):
        payload["state_dict"] = policy.state_dict()
    if hasattr(policy, "obs_dim"):
        payload["obs_dim"] = int(policy.obs_dim)
    if hasattr(policy, "action_dim"):
        payload["action_dim"] = int(policy.action_dim)
    if hasattr(policy, "hidden_sizes"):
        payload["hidden_sizes"] = list(policy.hidden_sizes)
    if hasattr(policy, "discrete"):
        payload["discrete"] = bool(policy.discrete)
    if hasattr(policy, "log_std"):
        payload["log_std"] = np.asarray(policy.log_std.detach().cpu().numpy(), dtype=np.float32)
    return payload


def save_policy(
    policy: Any,
    path: str,
    env_id: Optional[Any] = None,
    kind: str = "policy",
    extra: Optional[Dict[str, Any]] = None,
    **meta: Any,
) -> str:
    """Persist a policy checkpoint (``torch.save`` of a metadata dict)."""
    if not PYTORCH_AVAILABLE:  # pragma: no cover
        raise RuntimeError("PyTorch is required to save policies.")
    path = str(path)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = _checkpoint_payload(policy)
    payload["env_key"] = normalize_env_key(env_id) if env_id is not None else getattr(policy, "env_key", None)
    payload["kind"] = str(kind)
    payload["meta"] = dict(meta)
    if extra:
        payload["meta"].update(dict(extra))
    torch.save(payload, path)
    return path


def load_policy(
    path: str,
    env_id: Optional[Any] = None,
    obs_dim: Optional[int] = None,
    action_dim: Optional[int] = None,
    action_space: Any = None,
    observation_space: Any = None,
    kind: Optional[str] = None,
    device: Any = "cpu",
    build: bool = True,
    **kwargs: Any,
):
    """Load a checkpoint written by :func:`save_policy`.

    With ``build=True`` (default) the module is reconstructed and returned;
    otherwise the raw checkpoint dict is returned.
    """
    if not PYTORCH_AVAILABLE:  # pragma: no cover
        raise RuntimeError("PyTorch is required to load policies.")
    payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, dict) or "state_dict" not in payload:
        if build:
            raise ValueError("Checkpoint %s does not contain a 'state_dict'." % path)
        return payload
    if not build:
        return payload

    key = normalize_env_key(env_id if env_id is not None else payload.get("env_key"))
    kind = kind or payload.get("kind", "policy")
    is_mask = str(kind).lower().startswith("mask")
    spec = ARCHS[key]

    obs_dim = obs_dim if obs_dim is not None else (
        payload.get("obs_dim") or _obs_dim_from_space(observation_space) or spec["obs_dim"]
    )
    if is_mask:
        action_dim = int(payload.get("action_dim") or kwargs.pop("mask_logits", 2))
        discrete = True
    else:
        action_dim = action_dim if action_dim is not None else payload.get("action_dim")
    hidden_sizes = payload.get("hidden_sizes") or (
        spec["mask_hidden"] if is_mask else spec["policy_hidden"]
    )

    model = build_policy(
        env_id=key,
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_space=action_space,
        observation_space=observation_space,
        kind=kind,
        backend=kwargs.pop("backend", "native"),
        hidden_sizes=hidden_sizes,
        activation=kwargs.pop("activation", spec["activation"]),
        discrete=discrete if is_mask else payload.get("discrete"),
        device=device,
        **kwargs,
    )
    try:
        model.load_state_dict(payload["state_dict"], strict=True)
    except Exception:
        model.load_state_dict(payload["state_dict"], strict=False)
    try:
        model.eval()
    except Exception:  # pragma: no cover
        pass
    return model
