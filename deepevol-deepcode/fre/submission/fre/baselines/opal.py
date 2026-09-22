"""OPAL baseline (Ajay et al., 2020) re-implemented inside the FRE codebase.

Paper specification reproduced here (verbatim summary of the sources used):

* Section 5.2: "OPAL (Ajay et al., 2020), a representative offline unsupervised skill
  discovery method where latent skills are learned by auto-encoding trajectories."
  ... "OPAL is re-implemented in our codebase."  ... "Since OPAL does not solve the problem
  of understanding a reward function zero-shot, we compare to a version with privileged
  execution based on online rollouts."

* Addendum, "Additional Details on OPAL":
    - No manually designed rewards are used in OPAL.
    - For the OPAL encoder, the same transformer architecture is used as in FRE.
    - For the privileged execution evaluation described in the paper:
        * OPAL's task policy is not used,
        * 10 random skills are sampled from a unit Gaussian,
        * for each skill ``z``, the policy is conditioned on it and evaluated for the
          entire episode,
        * and the best performing rollout is taken.

Therefore this module provides

1. ``OPALEncoder`` -- a trajectory encoder built out of the *same* transformer blocks used by
   FRE (``fre.models.fre_encoder.TransformerBlock``): token dim 128, 4 attention heads,
   4 layers, MLP width 256, mean pooling over the trajectory tokens (no positional encodings,
   exactly like FRE).  It outputs the mean/std of a diagonal Gaussian ``q(z | tau)``.
2. ``TrajectoryDecoder`` -- auxiliary state reconstruction head used for the trajectory
   auto-encoding objective.
3. ``SkillPolicy`` -- tanh-squashed Gaussian skill-conditioned policy ``pi(a | s, z)``.
4. ``OPALAgent`` -- bundles encoder/decoder/policy with the information-theoretic objective
   (InfoNCE between sampled skills and trajectory encodings, in the spirit of OPAL's
   information asymmetry objective) plus KL regularization to the unit Gaussian prior
   (so that unit-Gaussian skills are on-manifold for privileged evaluation) and an
   auxiliary reconstruction term.  Rewards are *never* used for representation learning,
   matching "no manually designed rewards are used in OPAL".
5. ``evaluate_opal_privileged`` -- the privileged zero-shot-free evaluation protocol above
   (10 unit-Gaussian skills, whole-episode rollout each, keep the best), wired into the
   shared FRE evaluation harness so numbers are directly comparable to Table 1.
6. ``train_opal`` / ``main`` -- training loop and CLI, mirroring ``gc_iql``/``gc_bc``.

Torch is imported defensively so that the pure-data utilities remain importable in minimal
environments; numpy is required.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _HAS_TORCH = False


if _HAS_TORCH:

    _ModuleBase = nn.Module
else:  # pragma: no cover - torch is required in practice

    class _ModuleBase:  # minimal duck-typed stand-in so the module still imports
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
            raise RuntimeError("OPAL requires PyTorch to be installed.")


__all__ = [
    "OPALConfig",
    "OPALEncoder",
    "TrajectoryDecoder",
    "SkillPolicy",
    "OPALAgent",
    "OPALHistory",
    "info_nce_loss",
    "trajectory_iae_loss",
    "sample_trajectory_segments",
    "build_opal",
    "train_opal",
    "opal_action_fn",
    "skill_action_fn",
    "evaluate_opal_privileged",
    "evaluate_opal",
    "build_arg_parser",
    "main",
    "DEFAULT_OPAL_SKILL_DIM",
    "DEFAULT_OPAL_NUM_SKILLS",
    "DEFAULT_OPAL_SEGMENT_LENGTH",
    "DEFAULT_OPAL_HIDDEN_DIMS",
    "DEFAULT_OPAL_LEARNING_RATE",
    "DEFAULT_OPAL_BATCH_SIZE",
    "DEFAULT_OPAL_TRAIN_STEPS",
    "DEFAULT_OPAL_NUM_EPISODES",
    "DEFAULT_OPAL_NUM_SEEDS",
]


# --------------------------------------------------------------------------------------
# Defaults (Appendix A Table 3 architecture + Section 5.2 evaluation protocol)
# --------------------------------------------------------------------------------------
DEFAULT_OPAL_SKILL_DIM = 128  # same latent width as FRE's z
DEFAULT_OPAL_TOKEN_DIM = 128  # "same transformer architecture as in FRE"
DEFAULT_OPAL_LAYERS = 4
DEFAULT_OPAL_HEADS = 4
DEFAULT_OPAL_MLP_DIM = 256
DEFAULT_OPAL_HIDDEN_DIMS = (512, 512, 512)
DEFAULT_OPAL_LEARNING_RATE = 1e-4
DEFAULT_OPAL_BATCH_SIZE = 512
DEFAULT_OPAL_SEGMENT_LENGTH = 64
DEFAULT_OPAL_TRAIN_STEPS = 1_000_000
DEFAULT_OPAL_KL_COEF = 1.0
DEFAULT_OPAL_REC_COEF = 1.0
DEFAULT_OPAL_TEMPERATURE = 1.0
DEFAULT_OPAL_NUM_SKILLS = 10  # Addendum: "10 random skills are sampled from a unit Gaussian"
DEFAULT_OPAL_NUM_EPISODES = 20  # Section 5.2: "a mean over twenty evaluation episodes"
DEFAULT_OPAL_NUM_SEEDS = 5  # Section 5.2: "each agent is trained using five random seeds"
DEFAULT_OPAL_LOG_STD_MIN = -10.0
DEFAULT_OPAL_LOG_STD_MAX = 2.0
DEFAULT_OPAL_MAX_TRAJECTORY_LENGTH = 2048


# --------------------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------------------
def info_nce_loss(
    skills: Any,
    encodings: Any,
    temperature: float = DEFAULT_OPAL_TEMPERATURE,
    detach_targets: bool = False,
    eps: float = 1e-8,
) -> Any:
    """Information-asymmetry objective of OPAL (InfoNCE form).

    ``skills`` are the per-segment latent samples ``z ~ q(z | tau)`` and ``encodings`` the
    (mean) trajectory encodings returned by the posterior.  Scores are the negative squared
    Euclidean distance ``-|| z_i - z_hat_j ||^2 / temperature``; the diagonal is the
    positive pair:

        L = -1/B sum_i log softmax_j( -|| z_i - z_hat_j ||^2 / temperature )

    Args:
        skills: tensor ``(B, D)``.
        encodings: tensor ``(B, D)``.
        temperature: softmax temperature (default 1.0).
        detach_targets: detach ``encodings`` (stop-gradient targets).
        eps: unused numerical guard, kept for signature stability.

    Returns:
        Scalar tensor: the mean InfoNCE loss over the batch.
    """
    if not _HAS_TORCH:  # pragma: no cover
        raise RuntimeError("OPAL requires PyTorch.")
    targets = encodings.detach() if detach_targets else encodings
    # squared euclidean distances between all skill / encoding pairs
    dist = torch.cdist(skills, targets, p=2).pow(2)
    logits = -dist / max(float(temperature), eps)
    labels = torch.arange(skills.shape[0], device=skills.device, dtype=torch.long)
    return F.cross_entropy(logits, labels)


# Convenience alias matching OPAL terminology (information asymmetry estimator).
trajectory_iae_loss = info_nce_loss


def reconstruction_loss(predicted: Any, target: Any, mask: Any = None) -> Any:
    """Mean squared error used by the auxiliary trajectory auto-encoding term."""
    if not _HAS_TORCH:  # pragma: no cover
        raise RuntimeError("OPAL requires PyTorch.")
    err = (predicted - target) ** 2
    if mask is None:
        return err.mean()
    mask = mask.to(err.dtype)
    while mask.dim() < err.dim():
        mask = mask.unsqueeze(-1)
    denom = mask.sum().clamp_min(1.0)
    return (err * mask).sum() / denom


# --------------------------------------------------------------------------------------
# Networks
# --------------------------------------------------------------------------------------
class OPALEncoder(_ModuleBase):
    """Trajectory encoder built from the *same* transformer blocks used by FRE.

    Tokens are a linear projection of the raw trajectory states to ``token_dim`` (128 by
    default).  No positional encodings and no causal masking are applied, exactly matching
    FRE's permutation-invariant transformer (the Addendum states that OPAL's encoder uses
    "the same transformer architecture ... as in FRE").  Final-layer tokens are mean pooled
    and mapped by two linear heads to the mean and log-std of a diagonal Gaussian
    ``q(z | tau)``.
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int = DEFAULT_OPAL_SKILL_DIM,
        token_dim: int = DEFAULT_OPAL_TOKEN_DIM,
        num_layers: int = DEFAULT_OPAL_LAYERS,
        num_heads: int = DEFAULT_OPAL_HEADS,
        mlp_dim: int = DEFAULT_OPAL_MLP_DIM,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = True,
        log_std_min: float = DEFAULT_OPAL_LOG_STD_MIN,
        log_std_max: float = DEFAULT_OPAL_LOG_STD_MAX,
        use_positional_encoding: bool = False,
        max_trajectory_length: int = DEFAULT_OPAL_MAX_TRAJECTORY_LENGTH,
        name: str = "opal-encoder",
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.latent_dim = int(latent_dim)
        self.token_dim = int(token_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.use_positional_encoding = bool(use_positional_encoding)
        self.name = name

        self.state_proj = nn.Linear(self.state_dim, self.token_dim)
        if self.use_positional_encoding:
            self.position_embedding = nn.Embedding(int(max_trajectory_length), self.token_dim)
        else:
            self.position_embedding = None

        from fre.models.fre_encoder import TransformerBlock

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    token_dim=self.token_dim,
                    num_heads=int(num_heads),
                    mlp_dim=int(mlp_dim),
                    dropout=float(dropout),
                    activation=activation,
                    norm_first=bool(norm_first),
                )
                for _ in range(int(num_layers))
            ]
        )
        self.final_norm = nn.LayerNorm(self.token_dim)
        self.mean_head = nn.Linear(self.token_dim, self.latent_dim)
        self.log_std_head = nn.Linear(self.token_dim, self.latent_dim)
        self._init_parameters()

    # -- helpers ----------------------------------------------------------------------
    def _init_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def build_tokens(self, states: Any, mask: Any = None) -> Any:
        """``(B, T, state_dim)`` -> ``(B, T, token_dim)`` token embeddings."""
        if states.dim() == 2:
            states = states.unsqueeze(1)
        tokens = self.state_proj(states)
        if self.position_embedding is not None:
            timesteps = torch.arange(states.shape[1], device=states.device)
            tokens = tokens + self.position_embedding(timesteps).unsqueeze(0)
        if mask is not None:
            tokens = tokens * mask.unsqueeze(-1).to(tokens.dtype)
        return tokens

    def forward(
        self,
        states: Any,
        mask: Any = None,
        sample: bool = False,
        return_std: bool = False,
        return_tokens: bool = False,
    ) -> Dict[str, Any]:
        """Encode a batch of trajectories.

        Args:
            states: ``(B, T, state_dim)`` (``(B, state_dim)`` is treated as ``T = 1``).
            mask: optional ``(B, T)`` boolean mask, ``True`` = valid timestep.
            sample: if ``True`` reparameterize; otherwise return the posterior mean.
            return_std: also return the posterior std.
            return_tokens: also return pooled token features.

        Returns:
            Dict with ``mean``, ``log_std``/``std``, ``z`` (and optionally ``tokens``).
        """
        if not _HAS_TORCH:  # pragma: no cover
            raise RuntimeError("OPAL requires PyTorch.")
        if states.dim() == 2:
            states = states.unsqueeze(1)
        tokens = self.build_tokens(states, mask=mask)

        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask.bool()
            if key_padding_mask.shape != tokens.shape[:2]:
                key_padding_mask = key_padding_mask.reshape(tokens.shape[0], tokens.shape[1])

        for block in self.blocks:
            tokens = block(tokens, key_padding_mask=key_padding_mask)
            if mask is not None:
                tokens = tokens * mask.unsqueeze(-1).to(tokens.dtype)
        tokens = self.final_norm(tokens)

        if mask is not None:
            weights = mask.unsqueeze(-1).to(tokens.dtype)
            pooled = (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        else:
            pooled = tokens.mean(dim=1)

        mean = self.mean_head(pooled)
        log_std = self.log_std_head(pooled).clamp(self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        if sample:
            z = mean + std * torch.randn_like(std)
        else:
            z = mean

        out: Dict[str, Any] = {"mean": mean, "log_std": log_std, "std": std, "z": z}
        if return_tokens:
            out["tokens"] = tokens
        if not return_std:
            out.pop("std")
        return out

    def encode(self, states: Any, mask: Any = None) -> Any:
        """Posterior mean of ``z`` (used at evaluation time, like FRE)."""
        return self.forward(states, mask=mask, sample=False)["mean"]

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"state_dim={self.state_dim}, latent_dim={self.latent_dim}, "
            f"token_dim={self.token_dim}, layers={len(self.blocks)}, "
            f"pos_enc={self.use_positional_encoding}"
        )


class TrajectoryDecoder(_ModuleBase):
    """Auxiliary state-reconstruction head ``p(s_t | z, t)`` for trajectory auto-encoding.

    The skill ``z`` is concatenated with a learned timestep embedding and fed through a
    feed-forward MLP (``[512, 512, 512]`` by default) that predicts the state at each
    timestep.
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int = DEFAULT_OPAL_SKILL_DIM,
        hidden_dims: Sequence[int] = DEFAULT_OPAL_HIDDEN_DIMS,
        time_embed_dim: int = 16,
        activation: str = "relu",
        use_layer_norm: bool = False,
        max_trajectory_length: int = DEFAULT_OPAL_MAX_TRAJECTORY_LENGTH,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.latent_dim = int(latent_dim)
        self.time_embed_dim = int(time_embed_dim)
        from fre.models.fre_decoder import build_mlp

        self.time_embedding = nn.Embedding(int(max_trajectory_length), self.time_embed_dim)
        self.net = build_mlp(
            input_dim=self.latent_dim + self.time_embed_dim,
            hidden_dims=tuple(int(h) for h in hidden_dims),
            output_dim=self.state_dim,
            activation=activation,
            output_activation=None,
            use_layer_norm=bool(use_layer_norm),
        )

    def forward(self, z: Any, timesteps: Any) -> Any:
        """``z``: ``(B, D)``; ``timesteps``: ``(B, T)`` -> predicted states ``(B, T, state_dim)``."""
        if not _HAS_TORCH:  # pragma: no cover
            raise RuntimeError("OPAL requires PyTorch.")
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if timesteps.dim() == 1:
            timesteps = timesteps.unsqueeze(0)
        z_expanded = z.unsqueeze(1).expand(-1, timesteps.shape[1], -1)
        time_features = self.time_embedding(timesteps)
        inputs = torch.cat([z_expanded, time_features], dim=-1)
        return self.net(inputs)


class SkillPolicy(_ModuleBase):
    """Tanh-squashed Gaussian skill-conditioned policy ``pi(a | s, z)``.

    ``z`` is simply concatenated to the observation (the same conditioning strategy as FRE's
    IQL networks).  A deterministic action is obtained from the distribution mean, which is
    what privileged evaluation rolls out.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = DEFAULT_OPAL_SKILL_DIM,
        hidden_dims: Sequence[int] = DEFAULT_OPAL_HIDDEN_DIMS,
        activation: str = "relu",
        use_layer_norm: bool = False,
        action_low: Any = -1.0,
        action_high: Any = 1.0,
        log_std_min: float = DEFAULT_OPAL_LOG_STD_MIN,
        log_std_max: float = DEFAULT_OPAL_LOG_STD_MAX,
        init_gain: float = 0.01,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        from fre.models.fre_decoder import build_mlp

        self.net = build_mlp(
            input_dim=self.state_dim + self.latent_dim,
            hidden_dims=tuple(int(h) for h in hidden_dims),
            output_dim=2 * self.action_dim,
            activation=activation,
            output_activation=None,
            use_layer_norm=bool(use_layer_norm),
        )
        # small init on the output head for stable log-std
        last = None
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                last = module
        if isinstance(last, nn.Linear) and init_gain:
            nn.init.xavier_uniform_(last.weight, gain=float(init_gain))
            if last.bias is not None:
                nn.init.zeros_(last.bias)
            bias = torch.zeros(last.bias.shape)
            bias[: self.action_dim] = 0.0
            bias[self.action_dim :] = 0.0
            with torch.no_grad():
                last.bias.copy_(bias)

        low = np.asarray(action_low, dtype=np.float32)
        high = np.asarray(action_high, dtype=np.float32)
        self.register_buffer(
            "action_scale", torch.as_tensor((high - low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.as_tensor((high + low) / 2.0, dtype=torch.float32)
        )

    # -- internals --------------------------------------------------------------------
    def _compose(self, states: Any, skills: Any) -> Any:
        if states.dim() == 1:
            states = states.unsqueeze(0)
        if skills is None:
            skills = torch.zeros(
                states.shape[0], self.latent_dim, device=states.device, dtype=states.dtype
            )
        if skills.dim() == 1:
            skills = skills.unsqueeze(0).expand(states.shape[0], -1)
        if skills.dim() == 3:
            skills = skills.reshape(states.shape[0], -1)
        if skills.shape[0] == 1 and states.shape[0] > 1:
            skills = skills.expand(states.shape[0], -1)
        return torch.cat([states, skills.to(states.dtype)], dim=-1)

    def distribution(self, states: Any, skills: Any = None) -> Any:
        if not _HAS_TORCH:  # pragma: no cover
            raise RuntimeError("OPAL requires PyTorch.")
        outputs = self.net(self._compose(states, skills))
        mean, log_std = torch.split(outputs, self.action_dim, dim=-1)
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)
        return torch.distributions.Normal(mean, torch.exp(log_std))

    def mean_action(self, states: Any, skills: Any = None) -> Any:
        """Squashed deterministic action (distribution mean)."""
        dist = self.distribution(states, skills)
        action = torch.tanh(dist.mean)
        return action * self.action_scale + self.action_bias

    def forward(self, states: Any, skills: Any = None) -> Any:
        return self.mean_action(states, skills)

    def log_prob(self, states: Any, actions: Any, skills: Any = None) -> Any:
        """Log probability with the tanh change-of-variables correction."""
        dist = self.distribution(states, skills)
        actions = actions.to(dist.mean.dtype)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        if self.action_scale.numel() == 1 and self.action_scale.dim() == 0:
            scaled = (actions - self.action_bias) / self.action_scale.clamp_min(1e-6)
        else:
            scaled = (actions - self.action_bias.view(1, -1)) / self.action_scale.view(1, -1).clamp_min(
                1e-6
            )
        scaled = scaled.clamp(-0.999999, 0.999999)
        raw = torch.atanh(scaled)
        log_prob = dist.log_prob(raw)
        correction = torch.log(1.0 - scaled.pow(2) + 1e-6)
        log_prob = log_prob - correction
        if self.action_scale.dim() > 0:
            log_prob = log_prob - torch.log(self.action_scale.abs().clamp_min(1e-6)).sum(dim=-1)
        return log_prob.sum(dim=-1)

    @torch.no_grad() if _HAS_TORCH else (lambda fn: fn)  # type: ignore[misc]
    def act(self, states: Any, skills: Any = None, deterministic: bool = True) -> Any:
        if not _HAS_TORCH:  # pragma: no cover
            raise RuntimeError("OPAL requires PyTorch.")
        if deterministic:
            return self.mean_action(states, skills)
        dist = self.distribution(states, skills)
        action = torch.tanh(dist.rsample())
        return action * self.action_scale + self.action_bias


# --------------------------------------------------------------------------------------
# Data utilities
# --------------------------------------------------------------------------------------
def _as_state_array(dataset: Any) -> np.ndarray:
    """Best-effort extraction of an ``(N, state_dim)`` state array from an offline dataset."""
    if dataset is None:
        raise ValueError("A dataset is required to sample OPAL trajectory segments.")
    if isinstance(dataset, np.ndarray):
        return np.asarray(dataset, dtype=np.float32)
    for attr in ("observations", "states", "obs"):
        value = getattr(dataset, attr, None)
        if value is not None:
            return np.asarray(value, dtype=np.float32)
    for method in ("state_tensor", "states_tensor"):
        fn = getattr(dataset, method, None)
        if callable(fn):
            try:
                out = fn()
                if _HAS_TORCH and isinstance(out, torch.Tensor):
                    return out.detach().cpu().numpy().astype(np.float32)
                return np.asarray(out, dtype=np.float32)
            except Exception:
                continue
    raise TypeError(f"Could not extract states from dataset of type {type(dataset)!r}")


def trajectory_bounds(dataset: Any, num_transitions: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(traj_starts, traj_ends)`` (ends exclusive) for the dataset."""
    starts = getattr(dataset, "traj_starts", None)
    ends = getattr(dataset, "traj_ends", None)
    if starts is not None and ends is not None and len(starts) > 0:
        return np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64)
    traj_ids = getattr(dataset, "trajectory_ids", None)
    if traj_ids is not None and len(traj_ids) == num_transitions:
        traj_ids = np.asarray(traj_ids)
        boundaries = np.flatnonzero(np.diff(traj_ids) != 0) + 1
        starts = np.concatenate([[0], boundaries]).astype(np.int64)
        ends = np.concatenate([boundaries, [num_transitions]]).astype(np.int64)
        return starts, ends
    # fall back to a single (long) trajectory
    return np.array([0], dtype=np.int64), np.array([num_transitions], dtype=np.int64)


def sample_trajectory_segments(
    dataset: Any,
    batch_size: int = DEFAULT_OPAL_BATCH_SIZE,
    segment_length: int = DEFAULT_OPAL_SEGMENT_LENGTH,
    rng: Optional[np.random.Generator] = None,
    return_actions: bool = True,
) -> Dict[str, np.ndarray]:
    """Sample contiguous trajectory segments (the OPAL auto-encoding unit).

    Segments are drawn at random start positions inside randomly chosen trajectories, so the
    encoder sees ordered state sequences of length ``segment_length`` (padded/left-padded by
    repeating the first state when the trajectory is shorter than the segment length).

    Returns:
        Dict with ``states`` ``(B, T, state_dim)`` (and ``actions`` ``(B, T, action_dim)``
        when the dataset exposes actions).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    states = _as_state_array(dataset)
    num_transitions = states.shape[0]
    starts, ends = trajectory_bounds(dataset, num_transitions)
    num_trajs = len(starts)
    traj_idx = rng.integers(0, num_trajs, size=int(batch_size))
    segments = np.empty((int(batch_size), int(segment_length), states.shape[1]), dtype=np.float32)
    for i, t in enumerate(traj_idx):
        t0, t1 = int(starts[t]), int(ends[t])
        length = max(t1 - t0, 1)
        if length >= segment_length:
            offset = int(rng.integers(0, length - int(segment_length) + 1))
        else:
            offset = 0
        idx = np.clip(t0 + offset + np.arange(int(segment_length)), t0, t1 - 1)
        segments[i] = states[idx]
    out: Dict[str, np.ndarray] = {"states": segments}
    if return_actions:
        actions = getattr(dataset, "actions", None)
        if actions is not None:
            actions = np.asarray(actions, dtype=np.float32)
            act_segments = np.empty(
                (int(batch_size), int(segment_length), actions.shape[1]), dtype=np.float32
            )
            for i, t in enumerate(traj_idx):
                t0, t1 = int(starts[t]), int(ends[t])
                length = max(t1 - t0, 1)
                if length >= segment_length:
                    offset = int(rng.integers(0, length - int(segment_length) + 1))
                else:
                    offset = 0
                idx = np.clip(t0 + offset + np.arange(int(segment_length)), t0, t1 - 1)
                act_segments[i] = actions[idx]
            out["actions"] = act_segments
    return out


# --------------------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------------------
class OPALAgent(_ModuleBase):
    """OPAL agent: trajectory transformer encoder + auxiliary decoder + skill policy.

    The representation objective is

        L = L_InfoNCE(z, q(tau)) + kl_coef * KL(q(z|tau) || N(0, I)) + rec_coef * MSE(s, s_hat)

    with ``z ~ q(z | tau)`` the reparameterized skill.  The KL term makes unit-Gaussian
    skills on-manifold, which is what the privileged evaluation protocol relies on.  When
    actions are available the skill policy is trained by maximum likelihood
    ``-log pi(a_t | s_t, z)`` on the same segments (a behavior-cloning style "skill decoder"),
    keeping OPAL reward-free as specified by the Addendum.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = DEFAULT_OPAL_SKILL_DIM,
        token_dim: int = DEFAULT_OPAL_TOKEN_DIM,
        num_layers: int = DEFAULT_OPAL_LAYERS,
        num_heads: int = DEFAULT_OPAL_HEADS,
        mlp_dim: int = DEFAULT_OPAL_MLP_DIM,
        hidden_dims: Sequence[int] = DEFAULT_OPAL_HIDDEN_DIMS,
        learning_rate: float = DEFAULT_OPAL_LEARNING_RATE,
        policy_learning_rate: Optional[float] = None,
        kl_coef: float = DEFAULT_OPAL_KL_COEF,
        rec_coef: float = DEFAULT_OPAL_REC_COEF,
        temperature: float = DEFAULT_OPAL_TEMPERATURE,
        activation: str = "relu",
        log_std_min: float = DEFAULT_OPAL_LOG_STD_MIN,
        log_std_max: float = DEFAULT_OPAL_LOG_STD_MAX,
        action_low: Any = -1.0,
        action_high: Any = 1.0,
        use_positional_encoding: bool = False,
        max_trajectory_length: int = DEFAULT_OPAL_MAX_TRAJECTORY_LENGTH,
        device: Optional[Any] = None,
        name: str = "opal",
    ) -> None:
        super().__init__()
        if not _HAS_TORCH:
            raise RuntimeError("OPALAgent requires PyTorch.")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        self.kl_coef = float(kl_coef)
        self.rec_coef = float(rec_coef)
        self.temperature = float(temperature)
        self.learning_rate = float(learning_rate)
        self.name = name

        self.encoder = OPALEncoder(
            state_dim=self.state_dim,
            latent_dim=self.latent_dim,
            token_dim=token_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            activation="gelu",
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            use_positional_encoding=use_positional_encoding,
            max_trajectory_length=max_trajectory_length,
        )
        self.decoder = TrajectoryDecoder(
            state_dim=self.state_dim,
            latent_dim=self.latent_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            max_trajectory_length=max_trajectory_length,
        )
        self.policy = SkillPolicy(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            latent_dim=self.latent_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            action_low=action_low,
            action_high=action_high,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        )
        self.rep_optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.learning_rate,
        )
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=float(policy_learning_rate or learning_rate)
        )
        self.train_steps = 0
        if device is not None:
            self.to(device)

    # -- helpers ----------------------------------------------------------------------
    def device(self) -> Any:
        return next(self.parameters()).device

    def _to_tensor(self, array: Any, dtype: Any = None) -> Any:
        if isinstance(array, torch.Tensor):
            return array.to(self.device(), non_blocking=True)
        dtype = dtype or torch.float32
        return torch.as_tensor(np.asarray(array), dtype=dtype, device=self.device())

    # -- training ---------------------------------------------------------------------
    def update(
        self,
        batch: Mapping[str, Any],
        policy_update: bool = True,
        grad_clip: float = 10.0,
    ) -> Dict[str, float]:
        """One OPAL update on a batch of trajectory segments."""
        if not _HAS_TORCH:  # pragma: no cover
            raise RuntimeError("OPALAgent requires PyTorch.")
        states = self._to_tensor(batch["states"] if "states" in batch else batch["observations"])
        if states.dim() == 3 and states.shape[1] < states.shape[-1]:
            # tolerate (B, state_dim, T) layouts
            states = states.transpose(1, 2)
        num_segments, horizon = states.shape[0], states.shape[1]
        timesteps = torch.arange(horizon, device=self.device()).unsqueeze(0).expand(num_segments, -1)

        out = self.encoder(states, sample=True, return_std=True)
        z, mean, log_std = out["z"], out["mean"], out["log_std"]

        from fre.models.fre_encoder import kl_divergence_to_unit_gaussian

        kl = kl_divergence_to_unit_gaussian(mean, log_std).mean()
        encodings = self.encoder.encode(states)
        ia = info_nce_loss(z, encodings, temperature=self.temperature)

        predicted = self.decoder(z, timesteps)
        rec = reconstruction_loss(predicted, states)

        rep_loss = ia + self.kl_coef * kl + self.rec_coef * rec
        self.rep_optimizer.zero_grad(set_to_none=True)
        rep_loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.decoder.parameters()), grad_clip
            )
        self.rep_optimizer.step()

        metrics: Dict[str, float] = {
            "ia_loss": float(ia.detach().cpu()),
            "kl": float(kl.detach().cpu()),
            "reconstruction_loss": float(rec.detach().cpu()),
            "rep_loss": float(rep_loss.detach().cpu()),
            "z_std": float(z.std(dim=0).mean().detach().cpu()),
        }

        if policy_update and "actions" in batch and batch["actions"] is not None:
            actions = self._to_tensor(batch["actions"])
            if actions.dim() == 3 and actions.shape[1] < actions.shape[-1]:
                actions = actions.transpose(1, 2)
            flat_states = states.reshape(-1, self.state_dim)
            flat_actions = actions.reshape(-1, self.action_dim)
            z_expanded = z.unsqueeze(1).expand(-1, horizon, -1).reshape(-1, self.latent_dim)
            log_prob = self.policy.log_prob(flat_states, flat_actions, z_expanded)
            policy_loss = -log_prob.mean()
            self.policy_optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), grad_clip)
            self.policy_optimizer.step()
            metrics["policy_loss"] = float(policy_loss.detach().cpu())
            metrics["log_prob_mean"] = float(log_prob.mean().detach().cpu())

        self.train_steps += 1
        metrics["train_steps"] = float(self.train_steps)
        return metrics

    # aliases
    def train_step(self, batch: Mapping[str, Any], **kwargs: Any) -> Dict[str, float]:
        return self.update(batch, **kwargs)

    def update_from_batch(self, batch: Mapping[str, Any], **kwargs: Any) -> Dict[str, float]:
        return self.update(batch, **kwargs)

    # -- acting -----------------------------------------------------------------------
    def encode(self, states: Any) -> np.ndarray:
        """Posterior-mean encoding of trajectories/states as a numpy array."""
        if not _HAS_TORCH:  # pragma: no cover
            raise RuntimeError("OPALAgent requires PyTorch.")
        with torch.no_grad():
            tensor = self._to_tensor(states)
            z = self.encoder.encode(tensor)
        return z.detach().cpu().numpy()

    def select_action(
        self,
        states: Any,
        skills: Any = None,
        deterministic: bool = True,
    ) -> np.ndarray:
        """Deterministic (default) or sampled action for ``(state, skill)`` inputs."""
        with torch.no_grad():
            tensor = self._to_tensor(states)
            skill_tensor = None if skills is None else self._to_tensor(skills)
            action = self.policy.act(tensor, skill_tensor, deterministic=deterministic)
        return action.detach().cpu().numpy()

    act = select_action

    def state_dict_full(self) -> Dict[str, Any]:
        return {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "policy": self.policy.state_dict(),
            "train_steps": self.train_steps,
            "hparams": self.hparams(),
        }

    def load_state_dict_full(self, state: Mapping[str, Any], strict: bool = False) -> None:
        self.encoder.load_state_dict(state["encoder"], strict=strict)
        self.decoder.load_state_dict(state["decoder"], strict=strict)
        self.policy.load_state_dict(state["policy"], strict=strict)
        self.train_steps = int(state.get("train_steps", 0))

    def hparams(self) -> Dict[str, Any]:
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "latent_dim": self.latent_dim,
            "learning_rate": self.learning_rate,
            "kl_coef": self.kl_coef,
            "rec_coef": self.rec_coef,
            "temperature": self.temperature,
            "train_steps": self.train_steps,
        }

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"state_dim={self.state_dim}, action_dim={self.action_dim}, latent={self.latent_dim}"


# --------------------------------------------------------------------------------------
# Configuration / factories
# --------------------------------------------------------------------------------------
@dataclass
class OPALConfig:
    """Configuration for OPAL training + privileged evaluation."""

    domain: str = "antmaze"
    state_dim: Optional[int] = None
    action_dim: Optional[int] = None
    latent_dim: int = DEFAULT_OPAL_SKILL_DIM
    hidden_dims: Tuple[int, ...] = DEFAULT_OPAL_HIDDEN_DIMS
    learning_rate: float = DEFAULT_OPAL_LEARNING_RATE
    batch_size: int = DEFAULT_OPAL_BATCH_SIZE
    segment_length: int = DEFAULT_OPAL_SEGMENT_LENGTH
    train_steps: int = DEFAULT_OPAL_TRAIN_STEPS
    kl_coef: float = DEFAULT_OPAL_KL_COEF
    rec_coef: float = DEFAULT_OPAL_REC_COEF
    temperature: float = DEFAULT_OPAL_TEMPERATURE
    use_positional_encoding: bool = False
    log_std_min: float = DEFAULT_OPAL_LOG_STD_MIN
    log_std_max: float = DEFAULT_OPAL_LOG_STD_MAX
    action_low: Any = -1.0
    action_high: Any = 1.0
    # privileged evaluation (Addendum "Additional Details on OPAL")
    num_skills: int = DEFAULT_OPAL_NUM_SKILLS
    num_episodes: int = DEFAULT_OPAL_NUM_EPISODES
    seeds: Tuple[int, ...] = tuple(range(DEFAULT_OPAL_NUM_SEEDS))
    deterministic: bool = True
    discretize_antmaze: bool = True
    num_bins: int = 32
    max_episode_steps: Optional[int] = None
    device: Optional[str] = None
    seed: int = 0
    dataset_kwargs: Dict[str, Any] = field(default_factory=dict)

    def replace(self, **overrides: Any) -> "OPALConfig":
        values = copy.deepcopy(self.__dict__)
        values.update(overrides)
        return OPALConfig(**values)

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.__dict__)


def build_opal(
    state_dim: int,
    action_dim: int,
    latent_dim: int = DEFAULT_OPAL_SKILL_DIM,
    device: Optional[Any] = None,
    **kwargs: Any,
) -> OPALAgent:
    """Tolerant factory for :class:`OPALAgent`."""
    return OPALAgent(
        state_dim=int(state_dim),
        action_dim=int(action_dim),
        latent_dim=int(latent_dim),
        device=device,
        **kwargs,
    )


@dataclass
class OPALHistory:
    """Lightweight training history container (mirrors the GC baselines)."""

    steps: List[int] = field(default_factory=list)
    metrics: List[Dict[str, float]] = field(default_factory=list)

    def log(self, step: int, metrics: Mapping[str, float]) -> None:
        self.steps.append(int(step))
        self.metrics.append({k: float(v) for k, v in metrics.items()})

    def to_dict(self) -> Dict[str, Any]:
        return {"steps": list(self.steps), "metrics": list(self.metrics)}

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as handle:
            json.dump(self.to_dict(), handle, indent=2)
        return path

    def latest(self, key: str, default: Optional[float] = None) -> Optional[float]:
        for entry in reversed(self.metrics):
            if key in entry:
                return entry[key]
        return default


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
def train_opal(
    dataset: Any = None,
    config: Optional[OPALConfig] = None,
    steps: Optional[int] = None,
    device: Optional[Any] = None,
    agent: Optional[OPALAgent] = None,
    progress: bool = True,
    log_every: int = 5000,
    log_fn: Optional[Callable[[Mapping[str, float]], None]] = None,
    **config_overrides: Any,
) -> Dict[str, Any]:
    """Pretrain an OPAL agent on trajectory segments (no rewards used)."""
    if not _HAS_TORCH:
        raise RuntimeError("train_opal requires PyTorch.")
    config = config or OPALConfig()
    if config_overrides:
        config = config.replace(**config_overrides)

    if dataset is None:
        from fre.data import load_dataset

        dataset = load_dataset(config.domain, **config.dataset_kwargs)

    if config.state_dim is None:
        config = config.replace(state_dim=int(_as_state_array(dataset).shape[1]))

    start = time.time()
    rng = np.random.default_rng(config.seed)
    if agent is None:
        probe = sample_trajectory_segments(
            dataset,
            batch_size=2,
            segment_length=min(int(config.segment_length), 8),
            rng=rng,
        )
        action_dim = (
            int(probe["actions"].shape[-1]) if "actions" in probe else int(config.action_dim or 1)
        )
        agent = OPALAgent(
            state_dim=int(probe["states"].shape[-1]),
            action_dim=action_dim,
            latent_dim=int(config.latent_dim),
            hidden_dims=tuple(config.hidden_dims),
            learning_rate=float(config.learning_rate),
            kl_coef=float(config.kl_coef),
            rec_coef=float(config.rec_coef),
            temperature=float(config.temperature),
            log_std_min=float(config.log_std_min),
            log_std_max=float(config.log_std_max),
            action_low=config.action_low,
            action_high=config.action_high,
            use_positional_encoding=bool(config.use_positional_encoding),
            device=device or config.device,
        )

    total_steps = int(steps if steps is not None else config.train_steps)
    history = OPALHistory()
    iterator: Iterable[int]
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(range(total_steps), desc="opal", leave=False)
        except Exception:  # pragma: no cover
            iterator = range(total_steps)
    else:
        iterator = range(total_steps)

    final_metrics: Dict[str, float] = {}
    for step in iterator:
        batch = sample_trajectory_segments(
            dataset,
            batch_size=int(config.batch_size),
            segment_length=int(config.segment_length),
            rng=rng,
        )
        metrics = agent.update(batch)
        final_metrics = metrics
        if log_every and (step + 1) % int(log_every) == 0:
            history.log(step + 1, metrics)
            if log_fn is not None:
                log_fn(metrics)

    return {
        "agent": agent,
        "config": config,
        "steps": total_steps,
        "history": history,
        "final_metrics": final_metrics,
        "wall_time": time.time() - start,
    }


# --------------------------------------------------------------------------------------
# Privileged evaluation (Addendum "Additional Details on OPAL")
# --------------------------------------------------------------------------------------
def skill_action_fn(
    policy: Any,
    skill: Any,
    deterministic: bool = True,
    state_fn: Optional[Callable[[Any], Any]] = None,
) -> Callable[[Any], np.ndarray]:
    """Build an ``action_fn`` for :func:`fre.evaluation.evaluate.rollout_episode`.

    ``policy`` may be a :class:`SkillPolicy`/``nn.Module`` exposing ``act``/``select_action``
    or a plain callable ``policy(state, skill)``.
    """
    resolved_skill = None
    if skill is not None:
        resolved_skill = np.asarray(skill, dtype=np.float32).ravel()

    def action_fn(state: Any) -> np.ndarray:
        if state_fn is not None:
            state = state_fn(state)
        state_array = np.asarray(state, dtype=np.float32).ravel()
        if not _HAS_TORCH:
            raise RuntimeError("OPAL policy rollout requires PyTorch.")
        with torch.no_grad():
            state_tensor = torch.as_tensor(state_array, dtype=torch.float32).unsqueeze(0)
            skill_tensor = (
                None
                if resolved_skill is None
                else torch.as_tensor(resolved_skill, dtype=torch.float32).unsqueeze(0)
            )
            if hasattr(policy, "act"):
                action = policy.act(state_tensor, skill_tensor, deterministic=deterministic)
            elif hasattr(policy, "select_action"):
                action = policy.select_action(state_tensor, skill_tensor, deterministic=deterministic)
            elif hasattr(policy, "policy") and hasattr(policy.policy, "act"):
                action = policy.policy.act(state_tensor, skill_tensor, deterministic=deterministic)
            else:
                action = policy(state_array, resolved_skill)
                return np.asarray(action, dtype=np.float32).ravel()
        return action.detach().cpu().numpy().ravel()

    return action_fn


# Alias used by a few call sites / docs.
opal_action_fn = skill_action_fn


def _task_max_episode_steps(task: Any, default: Optional[int]) -> int:
    for attr in ("max_episode_steps",):
        value = getattr(task, attr, None)
        if isinstance(value, (int, np.integer)) and int(value) > 0:
            return int(value)
    metadata = getattr(task, "metadata", None)
    if isinstance(metadata, Mapping) and metadata.get("max_episode_steps"):
        return int(metadata["max_episode_steps"])
    return int(default or 1000)


def _call_supported(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call ``fn`` dropping kwargs its signature does not accept."""
    try:
        import inspect

        signature = inspect.signature(fn)
        accepts_var_kwargs = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
        )
        if accepts_var_kwargs:
            return fn(*args, **kwargs)
        filtered = {k: v for k, v in kwargs.items() if k in signature.parameters}
        return fn(*args, **filtered)
    except (TypeError, ValueError):
        return fn(*args, **kwargs)


def _make_task_env(
    domain: str,
    task: Any,
    seed: Optional[int] = None,
    max_episode_steps: Optional[int] = None,
    state_fn: Optional[Callable[[Any], Any]] = None,
) -> Any:
    """Create a task environment for the given domain (lazy, domain-specific imports)."""
    base = str(domain).split(":")[0]
    kwargs: Dict[str, Any] = {}
    if seed is not None:
        kwargs["seed"] = seed
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = max_episode_steps
    if state_fn is not None:
        kwargs["state_fn"] = state_fn

    if base == "antmaze":
        from fre.envs.antmaze_tasks import make_antmaze_task_env

        return _call_supported(make_antmaze_task_env, task, **kwargs)
    if base == "exorl":
        from fre.envs.exorl_tasks import make_exorl_task_env

        return _call_supported(make_exorl_task_env, task, **kwargs)
    if base == "kitchen":
        from fre.envs.kitchen_tasks import make_kitchen_task_env

        return _call_supported(make_kitchen_task_env, task, **kwargs)
    raise ValueError(f"Unknown domain {domain!r} for OPAL evaluation.")


def _observation_fn_for(domain: str, state_fn: Optional[Callable[[Any], Any]]) -> Any:
    """Build the observation transform used for policy inputs at eval time."""
    base = str(domain).split(":")[0]
    if state_fn is not None:
        return state_fn
    if base == "antmaze":
        try:
            from fre.data.preprocessing import discretize_antmaze_xy

            return lambda obs: discretize_antmaze_xy(
                np.asarray(obs, dtype=np.float32).reshape(1, -1)
            ).reshape(-1)
        except Exception:  # pragma: no cover - very defensive
            return None
    return None


def evaluate_opal_privileged(
    agent: OPALAgent,
    domain: str = "antmaze",
    task_set: str = "all",
    dataset: Any = None,
    num_episodes: int = DEFAULT_OPAL_NUM_EPISODES,
    seeds: Sequence[int] = tuple(range(DEFAULT_OPAL_NUM_SEEDS)),
    deterministic: bool = True,
    num_skills: int = DEFAULT_OPAL_NUM_SKILLS,
    max_episode_steps: Optional[int] = None,
    device: Optional[Any] = None,
    verbose: bool = False,
    calibration: Optional[Mapping[str, Any]] = None,
    rng: Optional[np.random.Generator] = None,
    **task_kwargs: Any,
) -> Dict[str, Any]:
    """Privileged execution evaluation of OPAL (Addendum "Additional Details on OPAL").

    For every evaluation episode: sample ``num_skills`` skills from a unit Gaussian, roll out
    the skill-conditioned policy for the *entire* episode for each skill, and keep the best
    performing rollout (score = the test task's return, which is privileged information not
    available offline).  Returns are normalized to ``[0, 100]`` and averaged over
    ``num_episodes`` episodes and ``seeds`` random seeds, matching Table 1.
    """
    from fre.evaluation.evaluate import (
        RolloutResult,
        normalize_episode_return,
        resolve_return_range,
        rollout_episode,
    )
    from fre.envs import build_tasks

    base_domain = str(domain).split(":")[0]
    tasks = build_tasks(domain, task_set=task_set, **task_kwargs) if not isinstance(
        task_set, (list, tuple)
    ) else list(task_set)
    if isinstance(task_set, (list, tuple)) and not tasks:
        tasks = list(task_set)

    generator = rng if rng is not None else np.random.default_rng(0)
    if device is not None and agent is not None:
        try:
            agent.to(device)
        except Exception:  # pragma: no cover
            pass

    results: Dict[str, Dict[str, Any]] = {}
    for task in tasks:
        name = str(getattr(task, "name", task))
        steps = _task_max_episode_steps(task, max_episode_steps)
        try:
            ref_min, ref_max = resolve_return_range(task, None, calibration)
        except Exception:  # pragma: no cover - defensive
            ref_min, ref_max = 0.0, 1.0

        observation_fn = _observation_fn_for(domain, None)
        seed_normalized: List[float] = []
        seed_raw: List[float] = []
        seed_success: List[float] = []

        for seed in seeds:
            episode_scores: List[float] = []
            episode_raw: List[float] = []
            episode_success: List[float] = []
            for ep in range(int(num_episodes)):
                # fresh unit-Gaussian skills for this episode
                skills = generator.standard_normal((int(num_skills), int(agent.latent_dim))).astype(
                    np.float32
                )
                best_raw = -np.inf
                best_success = 0.0
                for skill_index in range(int(num_skills)):
                    env = _make_task_env(
                        domain,
                        task,
                        seed=int(seed) * 1000 + ep,
                        max_episode_steps=steps,
                    )
                    try:
                        action_fn = skill_action_fn(
                            agent.policy if agent is not None else None,
                            skills[skill_index],
                            deterministic=deterministic,
                        )
                        rollout = rollout_episode(
                            env,
                            action_fn,
                            task,
                            max_episode_steps=steps,
                            seed=int(seed) * 1000 + ep,
                            observation_fn=observation_fn,
                        )
                        raw = float(getattr(rollout, "return_", 0.0))
                        success = bool(getattr(rollout, "success", False))
                    except Exception as exc:  # pragma: no cover - env availability
                        if verbose:
                            print(f"[opal] rollout failed for {name}: {exc}")
                        raw, success = 0.0, False
                    finally:
                        try:
                            env.close()
                        except Exception:
                            pass
                    if raw > best_raw:
                        best_raw = raw
                        best_success = 1.0 if success else 0.0
                if not np.isfinite(best_raw):
                    best_raw = 0.0
                episode_raw.append(float(best_raw))
                episode_success.append(float(best_success))
                episode_scores.append(
                    float(normalize_episode_return(best_raw, ref_min, ref_max, clip=True))
                )

            seed_raw.append(float(np.mean(episode_raw)) if episode_raw else 0.0)
            seed_normalized.append(float(np.mean(episode_scores)) if episode_scores else 0.0)
            seed_success.append(float(np.mean(episode_success)) if episode_success else 0.0)
            if verbose:
                print(
                    f"[opal] {name} seed={seed} normalized={seed_normalized[-1]:.1f} "
                    f"raw={seed_raw[-1]:.2f}"
                )

        mean_normalized = float(np.mean(seed_normalized)) if seed_normalized else 0.0
        std_normalized = float(np.std(seed_normalized)) if seed_normalized else 0.0
        results[name] = {
            "mean": mean_normalized,
            "std": std_normalized,
            "normalized": mean_normalized,
            "normalized_std": std_normalized,
            "raw_mean": float(np.mean(seed_raw)) if seed_raw else 0.0,
            "raw_std": float(np.std(seed_raw)) if seed_raw else 0.0,
            "success_rate": float(np.mean(seed_success)) if seed_success else 0.0,
            "num_seeds": int(len(seed_normalized)),
            "num_episodes": int(num_episodes),
            "num_skills": int(num_skills),
            "per_seed": [float(v) for v in seed_normalized],
            "ref_min": float(ref_min),
            "ref_max": float(ref_max),
        }

    task_means = [v["normalized"] for v in results.values()]
    summary = {
        "mean": float(np.mean(task_means)) if task_means else 0.0,
        "std": float(np.std(task_means)) if task_means else 0.0,
        "num_tasks": int(len(task_means)),
    }
    return {
        "method": "opal",
        "domain": domain,
        "task_set": task_set if isinstance(task_set, str) else "custom",
        "num_skills": int(num_skills),
        "num_episodes": int(num_episodes),
        "seeds": [int(s) for s in seeds],
        "privileged": True,
        "tasks": results,
        "results": results,
        "summary": summary,
        "aggregate": summary,
    }


# Alias
evaluate_opal = evaluate_opal_privileged


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate the OPAL baseline.")
    parser.add_argument("--domain", type=str, default="antmaze")
    parser.add_argument("--task-set", type=str, default="all")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_OPAL_BATCH_SIZE)
    parser.add_argument("--segment-length", type=int, default=DEFAULT_OPAL_SEGMENT_LENGTH)
    parser.add_argument("--skill-dim", type=int, default=DEFAULT_OPAL_SKILL_DIM)
    parser.add_argument("--lr", type=float, default=DEFAULT_OPAL_LEARNING_RATE)
    parser.add_argument("--num-skills", type=int, default=DEFAULT_OPAL_NUM_SKILLS)
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_OPAL_NUM_EPISODES)
    parser.add_argument("--eval-seeds", type=int, default=DEFAULT_OPAL_NUM_SEEDS)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--history", type=str, default=None)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--eval-out", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if not _HAS_TORCH:
        parser.error("OPAL requires PyTorch to be installed.")

    config = OPALConfig(
        domain=args.domain,
        latent_dim=args.skill_dim,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        segment_length=args.segment_length,
        num_skills=args.num_skills,
        num_episodes=args.eval_episodes,
        seeds=tuple(range(args.eval_seeds)),
        max_episode_steps=args.max_episode_steps,
        device=args.device,
        seed=args.seed,
    )

    from fre.data import load_dataset

    dataset = load_dataset(args.domain)
    states = _as_state_array(dataset)
    actions = getattr(dataset, "actions", None)
    if actions is None:
        parser.error("The dataset does not expose actions; OPAL requires offline transitions.")
    action_dim = int(np.asarray(actions).shape[1])

    agent = OPALAgent(
        state_dim=int(states.shape[1]),
        action_dim=action_dim,
        latent_dim=int(config.latent_dim),
        hidden_dims=tuple(config.hidden_dims),
        learning_rate=float(config.learning_rate),
        kl_coef=float(config.kl_coef),
        rec_coef=float(config.rec_coef),
        temperature=float(config.temperature),
        device=args.device,
    )

    if args.load and os.path.exists(args.load):
        checkpoint = torch.load(args.load, map_location="cpu")
        state = checkpoint.get("agent", checkpoint)
        agent.load_state_dict_full(state)

    history = OPALHistory()
    if not args.skip_train:
        out = train_opal(
            dataset=dataset,
            config=config,
            steps=args.steps,
            device=args.device,
            agent=agent,
            progress=True,
            log_every=5000,
        )
        history = out["history"]
        if args.history:
            history.save(args.history)

    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        torch.save(
            {"agent": agent.state_dict_full(), "config": config.to_dict()},
            args.save,
        )

    if args.no_eval:
        return 0

    summary = evaluate_opal_privileged(
        agent,
        domain=args.domain,
        task_set=args.task_set,
        dataset=dataset,
        num_episodes=int(config.num_episodes),
        seeds=tuple(config.seeds),
        num_skills=int(config.num_skills),
        max_episode_steps=config.max_episode_steps,
        device=args.device,
        verbose=bool(args.verbose),
    )
    if args.eval_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.eval_out)), exist_ok=True)
        with open(args.eval_out, "w") as handle:
            json.dump(summary, handle, indent=2)
    print(
        f"opal {args.domain}/{args.task_set}: "
        f"{summary['summary']['mean']:.1f} ± {summary['summary']['std']:.1f}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
