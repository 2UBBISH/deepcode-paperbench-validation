"""OPAL baseline for the FRE reproduction (Table 1: ``OPAL`` / ``OPAL-10``).

OPAL ("Off-Policy Adversarial Latent") learns a latent skill space from
*unsupervised* offline transitions and then trains a latent-conditioned policy
with off-policy RL.  At evaluation time OPAL is *privileged*: it samples a small
number of skills from the latent prior, rolls each one out, and keeps the best
rollout (the ``OPAL-10`` entry in the FRE paper uses 10 skills).

Design notes / reproduction choices
-----------------------------------
* The paper's reproduction plan explicitly states: *"OPAL: reuse FRE transformer
  encoder; privileged eval samples 10 Gaussian skills, picks best rollout."*
  We therefore instantiate :class:`~fre.fre.encoder.FREEncoder` as the skill
  encoder (a permutation-invariant transformer over a set of K=32
  ``(state, reward)`` context tokens).  For OPAL the context reward channel is
  optional: by default we feed a constant reward (``context_reward_mode="zero"``)
  so the skill posterior is inferred from states only, which matches OPAL's
  unsupervised state-transition inference.
* Skill inference is trained as a variational auto-encoder over *transitions*:
  the encoder infers ``q(z | context)`` and a decoder predicts the next state
  from ``(state, z)``.  The objective is ``MSE(next_state) + beta * KL`` — the
  same information-bottleneck form used for FRE (Eq. 6) with ``beta=0.01``.
* The latent-conditioned policy is trained with the shared IQL implementation
  from :mod:`fre.rl.iql` using the dataset rewards (``gamma=0.88``,
  ``expectile=0.8``, ``AWR temperature=3.0``, target update ``0.001``).
* OPAL's action-prior regularizer is implemented as an explicit KL penalty
  between the latent-conditioned policy and a behaviour-cloned action prior
  :math:`\\pi_{prior}(a|s)`.

The module is importable without torch (torch is soft-imported) and works both
as a library and as a CLI (``python -m fre.baselines.opal --domain antmaze``).
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional torch import (the module stays importable for numpy-only tooling)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised depending on environment
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:  # pragma: no cover
        raise ImportError(
            "OPAL requires PyTorch. Install torch>=1.13 to train/evaluate OPAL."
        )


# ---------------------------------------------------------------------------
# Defensive imports of shared FRE components
# ---------------------------------------------------------------------------
def _import_attr(candidates: Sequence[Tuple[str, str]], name: str) -> Any:
    """Import ``name`` from the first importable ``module`` in ``candidates``."""
    last_error: Optional[Exception] = None
    for module_name, attr in candidates:
        try:
            module = __import__(module_name, fromlist=[attr])
            if attr and hasattr(module, attr):
                return getattr(module, attr)
            if not attr:
                return module
        except Exception as exc:  # pragma: no cover - depends on import layout
            last_error = exc
            continue
    return _MISSING if last_error is not None else _MISSING


class _Missing:
    def __repr__(self) -> str:  # pragma: no cover
        return "<missing>"


_MISSING = _Missing()

FREEncoder = _import_attr(
    [
        ("fre.fre.encoder", "FREEncoder"),
        ("..fre.encoder", "FREEncoder"),
        ("fre.fre", "FREEncoder"),
    ],
    "FREEncoder",
)

LatentPolicyBundle = _import_attr(
    [
        ("fre.fre.latent_policy", "LatentPolicyBundle"),
        ("..fre.latent_policy", "LatentPolicyBundle"),
        ("fre.fre", "LatentPolicyBundle"),
    ],
    "LatentPolicyBundle",
)

IQLTrainer = _import_attr(
    [
        ("fre.rl.iql", "IQLTrainer"),
        ("..rl.iql", "IQLTrainer"),
        ("fre.rl", "IQLTrainer"),
    ],
    "IQLTrainer",
)

IQLConfig = _import_attr(
    [
        ("fre.rl.iql", "IQLConfig"),
        ("..rl.iql", "IQLConfig"),
        ("fre.rl", "IQLConfig"),
    ],
    "IQLConfig",
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_HIDDEN_SIZES: Tuple[int, ...] = (512, 512, 512)
DEFAULT_DECODER_HIDDEN: Tuple[int, ...] = (512, 512)
DEFAULT_LR = 1e-4
DEFAULT_BATCH_SIZE = 512
DEFAULT_DISCOUNT = 0.88
DEFAULT_EXPECTILE = 0.8
DEFAULT_AWR_TEMPERATURE = 3.0
DEFAULT_TARGET_UPDATE_RATE = 0.001
DEFAULT_CONTEXT_SIZE = 32
DEFAULT_LATENT_DIM = 128
DEFAULT_NUM_BLOCKS = 4
DEFAULT_NUM_HEADS = 4
DEFAULT_MLP_DIM = 256
DEFAULT_SKILL_BETA = 0.01
DEFAULT_ACTION_PRIOR_COEFF = 0.05
DEFAULT_NUM_EVAL_SKILLS = 10  # "OPAL-10" in the paper
DEFAULT_MAX_GRAD_NORM = 10.0
DEFAULT_STEPS = 1_000_000
DEFAULT_LOG_STD_MIN = -5.0
DEFAULT_LOG_STD_MAX = 2.0

_ACTIVATIONS: Dict[str, Any] = {}


def _activation(name: str) -> Any:
    if not _ACTIVATIONS:
        _ACTIVATIONS.update(
            {
                "relu": nn.ReLU,
                "gelu": nn.GELU,
                "tanh": nn.Tanh,
                "silu": nn.SiLU,
                "swish": nn.SiLU,
                "elu": nn.ELU,
                "mish": nn.Mish,
            }
        )
    return _ACTIVATIONS.get(str(name).lower(), nn.ReLU)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class OPALConfig:
    """Hyper-parameters for the OPAL baseline (FRE Table 1 comparators)."""

    # --- latent skill inference (reuses the FRE transformer encoder) --------
    latent_dim: int = DEFAULT_LATENT_DIM
    context_size: int = DEFAULT_CONTEXT_SIZE
    num_blocks: int = DEFAULT_NUM_BLOCKS
    num_heads: int = DEFAULT_NUM_HEADS
    mlp_dim: int = DEFAULT_MLP_DIM
    skill_beta: float = DEFAULT_SKILL_BETA
    context_reward_mode: str = "zero"  # {"zero", "batch"}
    decoder_hidden: Tuple[int, ...] = DEFAULT_DECODER_HIDDEN

    # --- latent-conditioned policy (IQL) ------------------------------------
    hidden_sizes: Tuple[int, ...] = DEFAULT_HIDDEN_SIZES
    lr: float = DEFAULT_LR
    batch_size: int = DEFAULT_BATCH_SIZE
    discount: float = DEFAULT_DISCOUNT
    expectile: float = DEFAULT_EXPECTILE
    awr_temperature: float = DEFAULT_AWR_TEMPERATURE
    target_update_rate: float = DEFAULT_TARGET_UPDATE_RATE
    bellman_target: str = "v"

    # --- OPAL action-prior regularizer --------------------------------------
    action_prior_coeff: float = DEFAULT_ACTION_PRIOR_COEFF
    action_prior_lr: Optional[float] = None
    log_std_min: float = DEFAULT_LOG_STD_MIN
    log_std_max: float = DEFAULT_LOG_STD_MAX

    # --- evaluation ---------------------------------------------------------
    num_eval_skills: int = DEFAULT_NUM_EVAL_SKILLS
    eval_episodes: int = 20
    eval_seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)

    # --- bookkeeping --------------------------------------------------------
    seed: int = 0
    device: str = "cpu"
    steps: int = DEFAULT_STEPS
    log_interval: int = 1000
    checkpoint_interval: int = 100_000
    output_dir: str = "./runs/opal"
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM
    latent_clip: Optional[float] = 5.0

    def as_dict(self) -> Dict[str, Any]:
        out = dict(self.__dict__)
        out["hidden_sizes"] = list(self.hidden_sizes)
        out["decoder_hidden"] = list(self.decoder_hidden)
        out["eval_seeds"] = list(self.eval_seeds)
        return out


# ---------------------------------------------------------------------------
# Fallback networks (used only when fre.fre.latent_policy is unavailable)
# ---------------------------------------------------------------------------
def _build_mlp(
    in_dim: int,
    hidden_sizes: Sequence[int],
    out_dim: int,
    activation: str = "relu",
    layer_norm: bool = False,
) -> Any:
    act_cls = _activation(activation)
    layers: List[Any] = []
    last = in_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(last, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act_cls())
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


if _TORCH_AVAILABLE:

    class _FallbackMLP(nn.Module):
        def __init__(
            self,
            in_dim: int,
            hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
            out_dim: int = 1,
            activation: str = "relu",
            layer_norm: bool = False,
        ) -> None:
            super().__init__()
            self.net = _build_mlp(in_dim, hidden_sizes, out_dim, activation, layer_norm)

        def forward(self, x: Any) -> Any:
            return self.net(x)

    class _FallbackGaussianPolicy(nn.Module):
        """Diagonal Gaussian policy with tanh-squashed actions."""

        def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            cond_dim: int = 0,
            hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
            activation: str = "relu",
            log_std_min: float = DEFAULT_LOG_STD_MIN,
            log_std_max: float = DEFAULT_LOG_STD_MAX,
        ) -> None:
            super().__init__()
            self.obs_dim = obs_dim
            self.action_dim = action_dim
            self.cond_dim = cond_dim
            self.log_std_min = log_std_min
            self.log_std_max = log_std_max
            self.body = _build_mlp(
                obs_dim + cond_dim, hidden_sizes, hidden_sizes[-1], activation, layer_norm=True
            )
            self.mean = nn.Linear(hidden_sizes[-1], action_dim)
            self.log_std = nn.Parameter(torch.full((action_dim,), log_std_min))

        def _cond(self, obs: Any, cond: Any = None) -> Any:
            if self.cond_dim <= 0:
                return obs
            if cond is None:
                raise ValueError("conditional policy requires a conditioning vector")
            if cond.dim() == 3:  # (B, A, Z) broadcasting for candidate actions
                obs = obs.unsqueeze(-2).expand(*cond.shape[:-1], obs.shape[-1])
            return torch.cat([obs, cond], dim=-1)

        def forward(self, obs: Any, cond: Any = None) -> Tuple[Any, Any]:
            h = self.body(self._cond(obs, cond))
            mean = self.mean(h)
            log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
            log_std = log_std.expand_as(mean)
            return mean, log_std

        def sample(self, obs: Any, cond: Any = None, deterministic: bool = False):
            mean, log_std = self.forward(obs, cond)
            if deterministic:
                return torch.tanh(mean), torch.zeros(mean.shape[:-1], device=mean.device)
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            x = normal.rsample()
            action = torch.tanh(x)
            log_prob = normal.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
            return action, log_prob.sum(-1)

        def log_prob(self, obs: Any, action: Any, cond: Any = None):
            mean, log_std = self.forward(obs, cond)
            eps = 1e-6
            action = action.clamp(-1 + eps, 1 - eps)
            x = torch.atanh(action)
            normal = torch.distributions.Normal(mean, log_std)
            lp = normal.log_prob(x) - torch.log(1 - action.pow(2) + eps)
            return lp.sum(-1)

        def act(self, obs: Any, cond: Any = None, deterministic: bool = True) -> Any:
            with torch.no_grad():
                action, _ = self.sample(obs, cond, deterministic=deterministic)
            return action


# ---------------------------------------------------------------------------
# Skill encoder (reuses the FRE transformer encoder)
# ---------------------------------------------------------------------------
class OPALSkillEncoder(nn.Module if _TORCH_AVAILABLE else object):  # type: ignore[misc]
    """Variational skill encoder ``q(z | context)``.

    Reuses :class:`fre.fre.encoder.FREEncoder` when available (the reproduction
    plan requires OPAL to reuse the FRE transformer encoder); otherwise falls
    back to a permutation-invariant mean-pool MLP over the context set.
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int = DEFAULT_LATENT_DIM,
        context_size: int = DEFAULT_CONTEXT_SIZE,
        num_blocks: int = DEFAULT_NUM_BLOCKS,
        num_heads: int = DEFAULT_NUM_HEADS,
        mlp_dim: int = DEFAULT_MLP_DIM,
        hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
    ) -> None:
        _require_torch()
        super().__init__()
        self.state_dim = int(state_dim)
        self.latent_dim = int(latent_dim)
        self.context_size = int(context_size)
        self.uses_fre_encoder = not isinstance(FREEncoder, _Missing)

        if self.uses_fre_encoder:
            self.encoder = FREEncoder(
                state_dim=self.state_dim,
                latent_dim=self.latent_dim,
                num_blocks=num_blocks,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
            )
        else:  # pragma: no cover - fallback path
            self.token = nn.Linear(self.state_dim, self.latent_dim)
            self.body = _build_mlp(
                self.latent_dim, tuple(hidden_sizes[:2]), self.latent_dim, "relu", True
            )
            self.mu_head = nn.Linear(self.latent_dim, self.latent_dim)
            self.log_sigma_head = nn.Linear(self.latent_dim, self.latent_dim)

    # -- helpers -----------------------------------------------------------
    def _reshape_context(self, context_states: Any) -> Any:
        if context_states.dim() == 2:
            context_states = context_states.unsqueeze(0)
        return context_states

    def forward(self, context_states: Any, context_rewards: Any = None):
        """Return ``(mu, log_sigma)`` for posterior ``q(z | context)``."""
        context_states = self._reshape_context(context_states)
        if self.uses_fre_encoder:
            if context_rewards is None:
                context_rewards = torch.zeros(
                    context_states.shape[:-1],
                    device=context_states.device,
                    dtype=context_states.dtype,
                )
            mu, log_sigma = self.encoder(context_states, context_rewards)
            return mu, log_sigma

        if context_rewards is not None and context_rewards.dim() == 3:
            context_rewards = context_rewards.squeeze(-1)
        tokens = self.body(self.token(context_states).mean(dim=-2))
        return self.mu_head(tokens), self.log_sigma_head(tokens).clamp(-10.0, 5.0)

    def sample(self, context_states: Any, context_rewards: Any = None):
        mu, log_sigma = self.forward(context_states, context_rewards)
        std = torch.exp(log_sigma)
        eps = torch.randn_like(std)
        return mu + std * eps, mu, log_sigma

    def encode(self, context_states: Any, context_rewards: Any = None, use_mean: bool = True):
        mu, log_sigma = self.forward(context_states, context_rewards)
        if use_mean:
            return mu
        return mu + torch.exp(log_sigma) * torch.randn_like(mu)


class SkillDecoder(nn.Module if _TORCH_AVAILABLE else object):  # type: ignore[misc]
    """Predicts the next state from ``(state, z)`` — the skill-VAE decoder."""

    def __init__(
        self,
        state_dim: int,
        latent_dim: int = DEFAULT_LATENT_DIM,
        hidden_sizes: Sequence[int] = DEFAULT_DECODER_HIDDEN,
    ) -> None:
        _require_torch()
        super().__init__()
        self.state_dim = int(state_dim)
        self.net = _build_mlp(state_dim + latent_dim, hidden_sizes, state_dim, "relu")

    def forward(self, states: Any, z: Any) -> Any:
        if states.dim() == 3:
            z = z.unsqueeze(-2).expand(*states.shape[:-1], z.shape[-1])
        return self.net(torch.cat([states, z], dim=-1))

    predict = forward


class ActionPrior(nn.Module if _TORCH_AVAILABLE else object):  # type: ignore[misc]
    """Behaviour-cloned Gaussian action prior ``pi_prior(a | s)`` (OPAL regularizer)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
        log_std_min: float = DEFAULT_LOG_STD_MIN,
        log_std_max: float = DEFAULT_LOG_STD_MAX,
    ) -> None:
        _require_torch()
        super().__init__()
        self.net = _build_mlp(obs_dim, hidden_sizes, hidden_sizes[-1], "relu", True)
        self.mean = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_min))
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, obs: Any):
        h = self.net(obs)
        mean = self.mean(h)
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max).expand_as(mean)
        return mean, log_std

    def log_prob(self, obs: Any, action: Any):
        mean, log_std = self.forward(obs)
        eps = 1e-6
        action = action.clamp(-1 + eps, 1 - eps)
        x = torch.atanh(action)
        normal = torch.distributions.Normal(mean, log_std)
        return (normal.log_prob(x) - torch.log(1 - action.pow(2) + eps)).sum(-1)

    def loss(self, obs: Any, action: Any):
        return -self.log_prob(obs, action).mean()


# ---------------------------------------------------------------------------
# Policy bundle access helpers (works with LatentPolicyBundle and fallbacks)
# ---------------------------------------------------------------------------
def _policy_module(bundle: Any) -> Any:
    for name in ("policy", "policy_network", "actor"):
        module = getattr(bundle, name, None)
        if module is not None and hasattr(module, "sample"):
            return module
    return bundle


def _skill_sample(bundle: Any, obs: Any, z: Any, deterministic: bool = False):
    module = _policy_module(bundle)
    try:
        return module.sample(obs, z, deterministic=deterministic)
    except TypeError:
        return module.sample(obs, z)


def _skill_log_prob(bundle: Any, obs: Any, action: Any, z: Any) -> Any:
    for name in ("policy_log_prob",):
        fn = getattr(bundle, name, None)
        if callable(fn):
            return fn(obs, action, z)
    module = _policy_module(bundle)
    try:
        return module.log_prob(obs, action, z)
    except TypeError:
        return module.log_prob(obs, z, action)


def _skill_act(bundle: Any, obs: Any, z: Any, deterministic: bool = True) -> Any:
    module = _policy_module(bundle)
    act = getattr(module, "act", None)
    if callable(act):
        try:
            return act(obs, z, deterministic=deterministic)
        except TypeError:
            return act(obs, z)
    action, _ = _skill_sample(bundle, obs, z, deterministic=deterministic)
    return action


def _soft_update_trainer(trainer: Any, rate: float) -> None:
    """Call ``soft_update`` when available, else fall back to hard updates."""
    fn = getattr(trainer, "soft_update", None)
    if callable(fn):
        try:
            fn(rate)
            return
        except TypeError:  # pragma: no cover - alternative signature
            fn()
            return
    # No soft-update support: emulate with an EMA over target params.
    q, qt = getattr(trainer, "q_network", None), getattr(trainer, "target_q_network", None)
    if q is not None and qt is not None:
        with torch.no_grad():
            for p, tp in zip(q.parameters(), qt.parameters()):
                tp.data.mul_(1.0 - rate).add_(rate * p.data)
    v, vt = getattr(trainer, "v_network", None), getattr(trainer, "target_v_network", None)
    if v is not None and vt is not None:
        with torch.no_grad():
            for p, tp in zip(v.parameters(), vt.parameters()):
                tp.data.mul_(1.0 - rate).add_(rate * p.data)


# ---------------------------------------------------------------------------
# OPAL agent
# ---------------------------------------------------------------------------
class OPALAgent:
    """OPAL baseline: latent skill VAE + z-conditioned IQL policy + action prior.

    Attributes
    ----------
    skill_encoder : OPALSkillEncoder
        Reuses the FRE transformer encoder to infer ``q(z | context)``.
    skill_decoder : SkillDecoder
        Predicts the next state given ``(state, z)`` (skill reconstruction).
    policy : LatentPolicyBundle
        z-conditioned Q/V/pi networks (trained by :class:`fre.rl.iql.IQLTrainer`).
    action_prior : ActionPrior
        Behaviour-cloned prior used by OPAL's action-prior KL regularizer.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: Optional[OPALConfig] = None,
        skill_state_dim: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        _require_torch()
        self.config = config or OPALConfig()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.skill_state_dim = int(skill_state_dim or obs_dim)
        self.device = torch.device(device or self.config.device or "cpu")

        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        latent_dim = self.config.latent_dim
        hidden = tuple(self.config.hidden_sizes)

        # --- skill VAE ----------------------------------------------------
        self.skill_encoder = OPALSkillEncoder(
            state_dim=self.skill_state_dim,
            latent_dim=latent_dim,
            context_size=self.config.context_size,
            num_blocks=self.config.num_blocks,
            num_heads=self.config.num_heads,
            mlp_dim=self.config.mlp_dim,
            hidden_sizes=hidden,
        ).to(self.device)
        self.skill_decoder = SkillDecoder(
            state_dim=self.skill_state_dim,
            latent_dim=latent_dim,
            hidden_sizes=tuple(self.config.decoder_hidden),
        ).to(self.device)

        # --- z-conditioned policy ----------------------------------------
        if not isinstance(LatentPolicyBundle, _Missing):
            self.policy = LatentPolicyBundle(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                latent_dim=latent_dim,
                hidden_sizes=hidden,
            ).to(self.device)
        else:  # pragma: no cover - fallback path
            policy = _FallbackGaussianPolicy(
                self.obs_dim, self.action_dim, latent_dim, hidden,
                log_std_min=self.config.log_std_min, log_std_max=self.config.log_std_max,
            )
            q_net = _FallbackMLP(self.obs_dim + self.action_dim + latent_dim, hidden, 1)
            v_net = _FallbackMLP(self.obs_dim + latent_dim, hidden, 1)
            self.policy = _FallbackPolicyBundle(policy, q_net, v_net).to(self.device)

        # --- action prior -------------------------------------------------
        self.action_prior = ActionPrior(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_sizes=hidden,
            log_std_min=self.config.log_std_min,
            log_std_max=self.config.log_std_max,
        ).to(self.device)

        # --- optimizers ---------------------------------------------------
        self.skill_optimizer = torch.optim.Adam(
            list(self.skill_encoder.parameters()) + list(self.skill_decoder.parameters()),
            lr=self.config.lr,
        )
        self.prior_optimizer = torch.optim.Adam(
            self.action_prior.parameters(),
            lr=self.config.action_prior_lr or self.config.lr,
        )

        # --- IQL trainer for the latent policy ----------------------------
        self.iql = None
        if not isinstance(IQLTrainer, _Missing):
            iql_config = self._make_iql_config()
            self.iql = IQLTrainer(
                policy=self.policy,
                encoder=None,
                reward_sampler=None,
                config=iql_config,
                device=str(self.device),
                latent_dim=latent_dim,
                lr=self.config.lr,
            )

        self.step = 0
        self._rng = np.random.default_rng(self.config.seed)

    # -- config plumbing ---------------------------------------------------
    def _make_iql_config(self) -> Any:
        if isinstance(IQLConfig, _Missing):  # pragma: no cover
            return None
        try:
            cfg = IQLConfig(
                discount=self.config.discount,
                expectile=self.config.expectile,
                awr_temperature=self.config.awr_temperature,
                target_update_rate=self.config.target_update_rate,
                lr=self.config.lr,
                batch_size=self.config.batch_size,
                bellman_target=self.config.bellman_target,
            )
        except TypeError:  # pragma: no cover - tolerate alternative constructors
            cfg = IQLConfig()
        return cfg

    # -- training ----------------------------------------------------------
    def _context_from_batch(self, batch: Dict[str, Any]):
        """Build ``(context_states, context_rewards)`` for skill inference."""
        dev = self.device
        states = batch.get("context_states")
        if states is not None:
            ctx = _as_tensor(states, dev)
            rewards = batch.get("context_rewards")
            if rewards is not None:
                rewards = _as_tensor(rewards, dev).reshape(ctx.shape[0], ctx.shape[1])
            return ctx, rewards

        obs = _as_tensor(batch["observations"], dev)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        B = obs.shape[0]
        K = min(self.config.context_size, B) if B > 0 else self.config.context_size
        if B == 0:  # pragma: no cover - degenerate batch
            ctx = torch.zeros(1, K, self.skill_state_dim, device=dev)
            return ctx, torch.zeros(1, K, device=dev)
        idx = torch.randint(0, B, (B, K), device=dev)
        ctx = obs[idx]  # (B, K, D)
        if ctx.shape[-1] != self.skill_state_dim:
            ctx = ctx[..., : self.skill_state_dim]
        if self.config.context_reward_mode == "batch" and "rewards" in batch:
            r = _as_tensor(batch["rewards"], dev).reshape(-1)
            rewards = r[idx].clamp(-1.0, 1.0)
        else:
            rewards = torch.zeros(B, K, device=dev)
        return ctx, rewards

    def skill_vae_loss(self, context_states: Any, context_rewards: Any, next_states: Any):
        z, mu, log_sigma = self.skill_encoder.sample(context_states, context_rewards)
        if self.config.latent_clip is not None:
            z = z.clamp(-self.config.latent_clip, self.config.latent_clip)
        pred_next = self.skill_decoder(next_states[:, 0, :], z)
        recon = F.mse_loss(pred_next, next_states[:, 0, :])
        kl = -0.5 * (1 + log_sigma - mu.pow(2) - log_sigma.exp()).sum(-1).mean()
        total = recon + self.config.skill_beta * kl
        return total, {"skill_recon": float(recon.detach()), "skill_kl": float(kl.detach())}

    def action_prior_loss(self, batch: Dict[str, Any]):
        obs = _as_tensor(batch["observations"], self.device)
        actions = _as_tensor(batch["actions"], self.device).clamp(-1 + 1e-6, 1 - 1e-6)
        return self.action_prior.loss(obs, actions)

    def action_prior_kl(self, z: Any, obs: Any) -> Any:
        """KL divergence between the skill policy and the action prior."""
        action, _ = _skill_sample(self.policy, obs, z, deterministic=False)
        log_pi = _skill_log_prob(self.policy, obs, action, z)
        log_prior = self.action_prior.log_prob(obs, action)
        return (log_pi - log_prior).mean()

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """One OPAL training step: skill VAE + action prior + IQL policy update."""
        self.step += 1
        metrics: Dict[str, float] = {}

        # Ensure tensor batch on device.
        batch = {k: _as_tensor(v, self.device) for k, v in batch.items() if v is not None}
        next_states = batch.get("next_observations")

        # --- 1. skill VAE --------------------------------------------------
        if next_states is not None:
            context_states, context_rewards = self._context_from_batch(batch)
            next_ctx = self._context_from_batch(
                {"observations": next_states}
                | (
                    {"context_states": batch["context_states"], "context_rewards": batch.get("context_rewards")}
                    if "context_states" in batch
                    else {}
                )
            )
            loss_skill, skill_metrics = self.skill_vae_loss(
                context_states, context_rewards, next_ctx[0]
            )
            self.skill_optimizer.zero_grad(set_to_none=True)
            loss_skill.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.skill_encoder.parameters()) + list(self.skill_decoder.parameters()),
                self.config.max_grad_norm,
            )
            self.skill_optimizer.step()
            metrics.update(skill_metrics)
            metrics["skill_loss"] = float(loss_skill.detach())

        # --- 2. action prior (behaviour cloning) ---------------------------
        loss_prior = self.action_prior_loss(batch)
        self.prior_optimizer.zero_grad(set_to_none=True)
        loss_prior.backward()
        torch.nn.utils.clip_grad_norm_(self.action_prior.parameters(), self.config.max_grad_norm)
        self.prior_optimizer.step()
        metrics["action_prior_loss"] = float(loss_prior.detach())

        # --- 3. latent policy (IQL on dataset rewards) ---------------------
        context_states, context_rewards = self._context_from_batch(batch)
        with torch.no_grad():
            z = self.skill_encoder.encode(context_states, context_rewards, use_mean=True)
            if self.config.latent_clip is not None:
                z = z.clamp(-self.config.latent_clip, self.config.latent_clip)
            if z.shape[0] != batch["observations"].shape[0]:
                z = z.expand(batch["observations"].shape[0], -1).contiguous()

        if self.iql is not None:
            iql_metrics = self.iql.update(batch, latents=z)
            metrics.update({f"iql_{k}" if not k.startswith("iql") else k: v
                            for k, v in dict(iql_metrics).items()})
            _soft_update_trainer(self.iql, self.config.target_update_rate)

        # --- 4. OPAL action-prior KL regularizer ---------------------------
        if self.config.action_prior_coeff > 0:
            obs = batch["observations"]
            kl = self.action_prior_kl(z.detach(), obs)
            policy_optimizer = self._policy_optimizer()
            if policy_optimizer is not None:
                self.policy.train()
                policy_optimizer.zero_grad(set_to_none=True)
                (self.config.action_prior_coeff * kl).backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.policy.parameters() if p.requires_grad],
                    self.config.max_grad_norm,
                )
                policy_optimizer.step()
                metrics["action_prior_kl"] = float(kl.detach())

        metrics["step"] = self.step
        return metrics

    def _policy_optimizer(self) -> Any:
        if self.iql is not None:
            for name in ("policy_optimizer", "pi_optimizer", "actor_optimizer"):
                opt = getattr(self.iql, name, None)
                if opt is not None:
                    return opt
        return None

    # -- acting ------------------------------------------------------------
    def sample_skills(
        self,
        num_skills: Optional[int] = None,
        generator: Any = None,
        mode: str = "prior",
    ) -> Any:
        """Sample skills. ``mode="prior"`` -> ``z ~ N(0, I)``."""
        n = int(num_skills or self.config.num_eval_skills)
        if generator is None:
            generator = torch.Generator(device=self.device).manual_seed(self.config.seed + self.step)
        shape = (n, self.config.latent_dim)
        if mode == "prior":
            z = torch.randn(shape, generator=generator, device=self.device)
        elif mode == "uniform":
            z = torch.rand(shape, generator=generator, device=self.device) * 2 - 1
        else:  # pragma: no cover - unsupported mode
            raise ValueError(f"unknown skill sampling mode: {mode!r}")
        if self.config.latent_clip is not None:
            z = z.clamp(-self.config.latent_clip, self.config.latent_clip)
        return z

    def select_action(self, obs: Any, z: Any = None, deterministic: bool = True) -> np.ndarray:
        """Act with a given skill (``z``); samples from the prior when ``None``."""
        self.policy.eval()
        obs_t = _as_tensor(obs, self.device).float()
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        if z is None:
            z = self.sample_skills(1)
        z_t = _as_tensor(z, self.device).float()
        if z_t.dim() == 1:
            z_t = z_t.unsqueeze(0)
        if z_t.shape[0] != obs_t.shape[0]:
            z_t = z_t.expand(obs_t.shape[0], -1).contiguous()
        with torch.no_grad():
            action = _skill_act(self.policy, obs_t, z_t, deterministic=deterministic)
        return action.cpu().numpy()

    act = select_action

    def encode_context(self, context_states: Any, context_rewards: Any = None, use_mean: bool = True) -> Any:
        """Encode a (state, reward) context into a skill latent ``z``."""
        with torch.no_grad():
            ctx = _as_tensor(context_states, self.device).float()
            rew = None if context_rewards is None else _as_tensor(context_rewards, self.device).float()
            return self.skill_encoder.encode(ctx, rew, use_mean=use_mean)

    # -- training modes / checkpointing ------------------------------------
    def train(self) -> "OPALAgent":
        self.skill_encoder.train()
        self.skill_decoder.train()
        self.action_prior.train()
        self.policy.train()
        return self

    def eval(self) -> "OPALAgent":
        self.skill_encoder.eval()
        self.skill_decoder.eval()
        self.action_prior.eval()
        self.policy.eval()
        return self

    def state_dict(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "skill_encoder": self.skill_encoder.state_dict(),
            "skill_decoder": self.skill_decoder.state_dict(),
            "action_prior": self.action_prior.state_dict(),
            "policy": self.policy.state_dict(),
            "skill_optimizer": self.skill_optimizer.state_dict(),
            "prior_optimizer": self.prior_optimizer.state_dict(),
            "step": self.step,
            "config": self.config.as_dict(),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "skill_state_dim": self.skill_state_dim,
        }
        if self.iql is not None:
            try:
                state["iql"] = self.iql.state_dict()
            except Exception:  # pragma: no cover
                pass
        return state

    def load_state_dict(self, state: Dict[str, Any], load_optimizer: bool = True) -> None:
        for key, module in (
            ("skill_encoder", self.skill_encoder),
            ("skill_decoder", self.skill_decoder),
            ("action_prior", self.action_prior),
            ("policy", self.policy),
        ):
            payload = state.get(key)
            if payload is not None:
                try:
                    module.load_state_dict(payload)
                except Exception:  # pragma: no cover - tolerate partial checkpoints
                    pass
        if load_optimizer:
            for key, opt in (
                ("skill_optimizer", self.skill_optimizer),
                ("prior_optimizer", self.prior_optimizer),
            ):
                payload = state.get(key)
                if payload is not None:
                    try:
                        opt.load_state_dict(payload)
                    except Exception:  # pragma: no cover
                        pass
        if self.iql is not None and "iql" in state:
            try:
                self.iql.load_state_dict(state["iql"], strict=False)
            except Exception:  # pragma: no cover
                pass
        self.step = int(state.get("step", self.step))

    def save(self, path: str) -> str:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(self.state_dict(), path)
        return path

    def load(self, path: str, load_optimizer: bool = True) -> "OPALAgent":
        state = torch.load(path, map_location=self.device)
        self.load_state_dict(state, load_optimizer=load_optimizer)
        return self

    # -- privileged evaluation --------------------------------------------
    def privileged_evaluate(
        self,
        env: Any,
        num_skills: Optional[int] = None,
        num_episodes: Optional[int] = None,
        seeds: Optional[Iterable[int]] = None,
        deterministic: bool = True,
        max_steps: Optional[int] = None,
        return_all: bool = False,
    ) -> Dict[str, Any]:
        """Sample ``num_skills`` skills, roll each out, and keep the best one.

        This mirrors the paper's privileged OPAL evaluation ("OPAL-10": sample 10
        Gaussian skills and pick the best rollout).
        """
        n_skills = int(num_skills or self.config.num_eval_skills)
        n_episodes = int(num_episodes or self.config.eval_episodes)
        seed_list = list(seeds if seeds is not None else self.config.eval_seeds)

        skills = self.sample_skills(n_skills)
        per_skill: List[float] = []
        per_skill_returns: List[float] = []

        for i in range(n_skills):
            z = skills[i : i + 1]
            returns, lengths, successes = [], [], []
            for seed in seed_list:
                for _ in range(max(n_episodes // max(len(seed_list), 1), 1)):
                    ret, length, success = self.rollout(
                        env, z, seed=seed, deterministic=deterministic, max_steps=max_steps
                    )
                    returns.append(ret)
                    lengths.append(length)
                    successes.append(success)
            per_skill.append(_score_rollouts(env, returns, lengths, successes))
            per_skill_returns.append(float(np.mean(returns)) if returns else 0.0)

        order = int(np.argmax(per_skill)) if per_skill else -1
        best = float(per_skill[order]) if per_skill else 0.0
        best_return = float(per_skill_returns[order]) if per_skill_returns else 0.0
        out: Dict[str, Any] = {
            "best_score": best,
            "best_skill": order,
            "best_return": best_return,
            "mean_score": float(np.mean(per_skill)) if per_skill else 0.0,
            "mean_return": float(np.mean(per_skill_returns)) if per_skill_returns else 0.0,
            "num_skills": n_skills,
            "num_episodes": n_episodes,
            "seeds": seed_list,
        }
        if return_all:
            out["per_skill_scores"] = per_skill
            out["per_skill_returns"] = per_skill_returns
            out["skills"] = skills.cpu().numpy()
        return out

    # -- rollout -----------------------------------------------------------
    def rollout(
        self,
        env: Any,
        z: Any = None,
        seed: int = 0,
        deterministic: bool = True,
        max_steps: Optional[int] = None,
    ) -> Tuple[float, int, float]:
        """Roll out a single episode conditioned on skill ``z``.

        Tolerates both 4- and 5-tuple ``step`` returns as well as heterogeneous
        ``reset`` signatures (mirrors :mod:`fre.evaluate`).
        """
        obs = _call_reset(env, seed)
        total, length, success = 0.0, 0, 0.0
        done = False
        limit = int(max_steps or getattr(env, "max_episode_steps", 1000) or 1000)
        while not done and length < limit:
            action = self.select_action(obs, z, deterministic=deterministic)
            step_out = env.step(np.asarray(action).reshape(-1))
            obs, reward, done, info = _unpack_step(step_out)
            total += float(reward)
            length += 1
            if isinstance(info, dict):
                for key in ("success", "is_success", "goal_reached", "subtask_success"):
                    if key in info and info[key] is not None:
                        success = max(success, float(bool(info[key])))
                        break
        last_success = getattr(env, "last_episode_success", None)
        if last_success is not None:
            success = max(success, float(bool(last_success)))
        return total, length, success


# ---------------------------------------------------------------------------
# Fallback policy bundle (only used when fre.fre.latent_policy is missing)
# ---------------------------------------------------------------------------
if _TORCH_AVAILABLE:

    class _FallbackPolicyBundle(nn.Module):
        def __init__(self, policy: Any, q_network: Any, v_network: Any) -> None:
            super().__init__()
            self.policy = policy
            self.policy_network = policy
            self.q_network = q_network
            self.v_network = v_network

        def q_value(self, obs: Any, action: Any, z: Any) -> Any:
            if action.dim() == 3:
                obs = obs.unsqueeze(-2).expand(*action.shape[:-1], obs.shape[-1])
                z = z.unsqueeze(-2).expand(*action.shape[:-1], z.shape[-1])
            elif z.dim() == 1 and obs.dim() > 1:
                z = z.unsqueeze(0).expand(obs.shape[0], -1)
            return self.q_network(torch.cat([obs, action, z], dim=-1)).squeeze(-1)

        def value(self, obs: Any, z: Any) -> Any:
            if z.dim() == 1 and obs.dim() > 1:
                z = z.unsqueeze(0).expand(obs.shape[0], -1)
            return self.v_network(torch.cat([obs, z], dim=-1)).squeeze(-1)

        def policy_log_prob(self, obs: Any, action: Any, z: Any) -> Any:
            return self.policy.log_prob(obs, action, z)


# ---------------------------------------------------------------------------
# Batch / env utilities
# ---------------------------------------------------------------------------
def _as_tensor(value: Any, device: Any) -> Any:
    if value is None:
        return None
    if _TORCH_AVAILABLE and isinstance(value, torch.Tensor):
        return value.to(device).float()
    if hasattr(value, "as_dict"):
        raise TypeError("expected an array, got a batch container")
    return torch.as_tensor(np.asarray(value), dtype=torch.float32, device=device)


def _call_reset(env: Any, seed: int) -> Any:
    for kwargs in (
        {"seed": seed},
        {"seed": seed, "goal": None},
        {"goal": None},
        {},
    ):
        try:
            out = env.reset(**kwargs)
        except TypeError:
            continue
        if isinstance(out, tuple) and len(out) == 2:
            return out[0]
        return out
    return env.reset()  # pragma: no cover


def _unpack_step(step_out: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
    if not isinstance(step_out, (tuple, list)):
        raise TypeError("env.step must return a tuple")
    if len(step_out) == 5:
        obs, reward, terminated, truncated, info = step_out
        done = bool(terminated) or bool(truncated)
    elif len(step_out) == 4:
        obs, reward, done, info = step_out
        done = bool(done)
    else:  # pragma: no cover
        raise ValueError(f"unexpected step tuple length: {len(step_out)}")
    return obs, float(np.asarray(reward).reshape(-1)[0]) if np.asarray(reward).size else 0.0, done, info or {}


def _score_rollouts(env: Any, returns: Sequence[float], lengths: Sequence[int], successes: Sequence[float]) -> float:
    """Map rollout statistics to the paper's normalized [0, 100] scale."""
    if not returns:
        return 0.0
    # Prefer explicit success information when the wrapper exposes it.
    if successes and any(s > 0 for s in successes):
        return float(100.0 * np.mean(successes))
    normalised = getattr(env, "normalized_score", None)
    if callable(normalised):
        try:
            return float(normalised(float(np.mean(successes)) if successes else 0.0))
        except Exception:  # pragma: no cover
            pass
    # Fall back to mapping the mean return onto [0, 100] using the environment's
    # reward range: sparse goal-reaching rewards live in [-1, 0], directional and
    # velocity rewards live in [-1, 1].
    max_steps = float(max(lengths) or getattr(env, "max_episode_steps", 1000) or 1000)
    mean_return = float(np.mean(returns))
    raw = getattr(env, "reward_range", None)
    if raw is not None:
        low, high = float(raw[0]), float(raw[1])
    else:
        low, high = -1.0, 0.0
    lo, hi = low * max_steps, high * max_steps
    if hi - lo < 1e-8:  # pragma: no cover
        return 0.0
    return float(np.clip(100.0 * (mean_return - lo) / (hi - lo), 0.0, 100.0))


# ---------------------------------------------------------------------------
# Builders / training loop
# ---------------------------------------------------------------------------
def build_opal(
    obs_dim: int,
    action_dim: int,
    config: Optional[OPALConfig] = None,
    skill_state_dim: Optional[int] = None,
    device: Optional[str] = None,
) -> OPALAgent:
    """Factory mirroring the other ``build_*`` helpers in the codebase."""
    return OPALAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=config,
        skill_state_dim=skill_state_dim,
        device=device,
    )


def _sample_buffer(batch_source: Any, batch_size: int, rng: np.random.Generator) -> Dict[str, Any]:
    """Sample a batch from a FRE replay buffer or a dataset dict."""
    if batch_source is None:
        raise ValueError("OPAL training requires a replay buffer or dataset dict")
    sample = getattr(batch_source, "sample", None)
    if callable(sample):
        try:
            batch = sample(batch_size, device="cpu")
        except TypeError:
            batch = sample(batch_size)
        if hasattr(batch, "as_dict"):
            batch = batch.as_dict()
        return dict(batch)

    # Plain dict of numpy arrays.
    data = dict(batch_source)
    n = len(np.asarray(data["observations"]))
    idx = rng.integers(0, n, size=min(batch_size, n))
    out: Dict[str, Any] = {}
    for key, value in data.items():
        try:
            out[key] = np.asarray(value)[idx]
        except Exception:  # pragma: no cover - non indexable fields
            continue
    return out


def train_opal(
    buffer: Any,
    obs_dim: int,
    action_dim: int,
    config: Optional[OPALConfig] = None,
    device: Optional[str] = None,
    steps: Optional[int] = None,
    logger: Any = None,
    agent: Optional[OPALAgent] = None,
    skill_state_dim: Optional[int] = None,
) -> OPALAgent:
    """Train OPAL on an offline dataset/replay buffer."""
    _require_torch()
    cfg = config or OPALConfig()
    total_steps = int(steps or cfg.steps)
    rng = np.random.default_rng(cfg.seed)

    if agent is None:
        agent = build_opal(obs_dim, action_dim, config=cfg, skill_state_dim=skill_state_dim, device=device)
    agent.train()

    start = time.time()
    last_metrics: Dict[str, float] = {}
    for step in range(1, total_steps + 1):
        batch = _sample_buffer(buffer, cfg.batch_size, rng)
        last_metrics = agent.update(batch)
        if step % cfg.log_interval == 0:
            msg = " | ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in sorted(last_metrics.items())
            )
            line = f"[opal] step {step}/{total_steps} ({time.time() - start:.1f}s) {msg}"
            if logger is not None and hasattr(logger, "info"):
                logger.info(line)
            else:
                print(line, flush=True)

    return agent


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the OPAL baseline (FRE Table 1).")
    parser.add_argument("--domain", default="antmaze", choices=["antmaze", "exorl", "kitchen"])
    parser.add_argument("--steps", type=int, default=None, help="gradient steps (default: 1e6)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--latent-dim", type=int, default=DEFAULT_LATENT_DIM)
    parser.add_argument("--context-size", type=int, default=DEFAULT_CONTEXT_SIZE)
    parser.add_argument("--num-eval-skills", type=int, default=DEFAULT_NUM_EVAL_SKILLS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="./runs/opal")
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--eval", action="store_true", help="run privileged 10-skill evaluation after training")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> OPALAgent:
    """CLI entry point: ``python -m fre.baselines.opal --domain antmaze``."""
    _require_torch()
    args = parse_args(argv)

    cfg = OPALConfig(
        latent_dim=args.latent_dim,
        context_size=args.context_size,
        lr=args.lr,
        batch_size=args.batch_size,
        num_eval_skills=args.num_eval_skills,
        seed=args.seed,
        device=args.device,
        steps=args.steps or DEFAULT_STEPS,
        log_interval=args.log_interval,
        output_dir=args.output_dir,
    )

    # --- load the offline dataset into a FRE replay buffer ----------------
    buffer = None
    obs_dim, action_dim = None, None
    if args.domain == "antmaze":
        from ..data.d4rl_loader import load_antmaze_buffer

        buffer = load_antmaze_buffer(trajectory_buffer=True)
        obs_dim = buffer.observations.shape[-1]
        action_dim = buffer.actions.shape[-1]
    elif args.domain == "kitchen":
        from ..data.d4rl_loader import load_kitchen_buffer

        buffer = load_kitchen_buffer(trajectory_buffer=True)
        obs_dim = buffer.observations.shape[-1]
        action_dim = buffer.actions.shape[-1]
    else:  # exorl
        from ..data.exorl_loader import load_exorl_dataset, to_replay_buffer

        dataset = load_exorl_dataset("walker")
        buffer = to_replay_buffer(dataset, trajectory_buffer=True)
        obs_dim = buffer.observations.shape[-1]
        action_dim = buffer.actions.shape[-1]

    agent = train_opal(buffer, obs_dim, action_dim, config=cfg, device=args.device)

    os.makedirs(cfg.output_dir, exist_ok=True)
    ckpt = os.path.join(cfg.output_dir, f"opal_{args.domain}_seed{args.seed}.pt")
    agent.save(ckpt)
    print(f"[opal] saved checkpoint to {ckpt}", flush=True)

    if args.eval:
        try:
            from ..evaluate import build_env, suite_for_domain
        except Exception:  # pragma: no cover - evaluation is optional
            print("[opal] evaluation utilities unavailable; skipping eval", flush=True)
            return agent
        domain = args.domain if args.domain != "exorl" else "exorl"
        suite = suite_for_domain(domain, dataset=None)
        for name, task in list(suite.items())[:3]:
            env = build_env(domain, task)
            result = agent.privileged_evaluate(env, num_skills=cfg.num_eval_skills)
            print(f"[opal] {name}: best_score={result['best_score']:.2f}", flush=True)
    return agent


# ---------------------------------------------------------------------------
# Aliases and exports
# ---------------------------------------------------------------------------
OPAL = OPALAgent
OPALAgentBaseline = OPALAgent
DEFAULT_NUM_SKILLS = DEFAULT_NUM_EVAL_SKILLS

__all__ = [
    "OPAL",
    "OPALAgent",
    "OPALAgentBaseline",
    "OPALConfig",
    "OPALSkillEncoder",
    "SkillDecoder",
    "ActionPrior",
    "build_opal",
    "train_opal",
    "parse_args",
    "main",
    "DEFAULT_LATENT_DIM",
    "DEFAULT_CONTEXT_SIZE",
    "DEFAULT_HIDDEN_SIZES",
    "DEFAULT_LR",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_DISCOUNT",
    "DEFAULT_EXPECTILE",
    "DEFAULT_AWR_TEMPERATURE",
    "DEFAULT_TARGET_UPDATE_RATE",
    "DEFAULT_SKILL_BETA",
    "DEFAULT_ACTION_PRIOR_COEFF",
    "DEFAULT_NUM_EVAL_SKILLS",
    "DEFAULT_NUM_SKILLS",
    "DEFAULT_STEPS",
]


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
