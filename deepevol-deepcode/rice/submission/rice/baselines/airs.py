"""AIRS explanation baseline for RICE (Stage-1 alternative explanation method).

This module implements the **AIRS** baseline (Yu et al., 2023) used in the RICE
paper (Cheng et al., ICML 2024) to answer the question *"does the refining stage
still help when the explanation is produced by a different method?"* (Sec. 4.1
Baseline Explanation Methods / Sec. 4.2 Experiment III / Appendix C.3, Table 6):

    "We investigate the impact of other explanation methods (i.e., AIRS
    (Yu et al., 2023) and Integrated Gradients (Sundararajan et al., 2017)) on
    four Mujoco games. We fix the refining method and use different explanation
    methods to identify critical steps for refinement. [...] Using other
    explanation methods (Integrated Gradients and AIRS), our framework still
    achieves better results than the random baseline, suggesting that our
    framework can work with different explanation method choices."

AIRS explains a (frozen) RL agent with an *attention mechanism*: a small
interpretable module learns which parts of the state (equivalently, which visited
steps) the agent's decision depends on, and those attention weights are the
explanation.  Concretely this module provides:

1. :class:`AIRSNetwork` -- a torch module that
   * projects the state into an embedding,
   * emits **feature attention** ``w(s) in Delta^d`` (softmax over state
     dimensions) and, when given a whole trajectory, **temporal attention**
     ``a_t in Delta^T`` (softmax over steps),
   * reconstructs the frozen policy's action output from the attention-masked
     state ``s_t (*) w(s_t)``.
   The training loss is the AIRS-style objective
   ``||F(s) - F_hat(s (*) w)||^2 + l1_coeff * mean(w)`` (action preservation plus
   a sparsity penalty on the attention), optionally with an entropy term that
   concentrates the temporal attention.

2. Step-level **importance scores** for arbitrary observations, obtained as the
   *attention-weighted per-dimension sensitivity* of the frozen policy

       I(s) = sum_j w_j(s) * |d F(s) / d s_j|

   where the sensitivity ``|dF/ds_j|`` is estimated with a robust
   finite-difference probe of the policy output (the frozen policies used by RICE
   are black boxes / SB3 MlpPolicies, so a zero-order estimate keeps the baseline
   usable with every backend).  Scores are optionally min-max normalised into
   ``[0, 1]`` so they are directly comparable to the mask network's ``P(keep)``.

3. A full drop-in explainer surface (:class:`AIRSScorer`,
   :class:`AIRSStateSelector`, :class:`AIRS`, :func:`make_airs`) mirroring
   :mod:`rice.baselines.random_explanation`,
   :mod:`rice.baselines.integrated_gradients` and
   :mod:`rice.explanation.importance` /
   :mod:`rice.explanation.critical_state`, so experiment drivers can swap
   ``{Random, StateMask, Integrated Gradients, AIRS, Ours}`` transparently.

Everything is defensively implemented: if torch, the frozen policy or the rest
of the ``rice`` package are unavailable, the module still imports and degrades to
plain finite-difference / deterministic-hashed importance scores.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional torch dependency (attention module + differentiable sensitivities)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised implicitly by the environment
    import torch
    import torch.nn as nn
    import torch.nn.functional as F  # noqa: N812

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# Optional project dependencies
# ---------------------------------------------------------------------------
try:  # pragma: no cover
    from ..utils.seeding import get_rng as _get_rng
except Exception:  # pragma: no cover

    def _get_rng(seed: Optional[int] = None):  # type: ignore[misc]
        return np.random.RandomState(seed) if seed is not None else np.random.RandomState()


try:  # pragma: no cover
    from ..utils.logging import get_logger as _get_logger
except Exception:  # pragma: no cover

    def _get_logger(name: str = "rice.baselines.airs", *_, **__):  # type: ignore[misc]
        return logging.getLogger(name)


try:  # pragma: no cover
    from ..models.policies import normalize_env_key as _normalize_env_key
except Exception:  # pragma: no cover

    def _normalize_env_key(env_id: Any) -> str:  # type: ignore[misc]
        if env_id is None:
            return "default"
        key = os.path.splitext(str(env_id).strip().lower())[0]
        for sep in ("-", " "):
            key = key.replace(sep, "_")
        parts = [p for p in key.split("_") if p]
        parts = [p for p in parts if not (p.startswith("v") and p[1:].isdigit())]
        return "_".join(parts) if parts else "default"


try:  # pragma: no cover
    from ..explanation.importance import (  # noqa: F401
        IMPORTANCE_MODES,
        TrajectoryImportance,
        extract_observations as _extract_observations,
    )

    _HAS_IMPORTANCE = True
except Exception:  # pragma: no cover
    _HAS_IMPORTANCE = False
    IMPORTANCE_MODES = ("mean", "max", "sum", "last", "first", "topk_mean")
    TrajectoryImportance = None  # type: ignore[assignment]
    _extract_observations = None  # type: ignore[assignment]


try:  # pragma: no cover
    from ..explanation.critical_state import (  # noqa: F401
        CriticalState,
        roll_trajectory as _roll_trajectory,
    )

    _HAS_CRITICAL = True
except Exception:  # pragma: no cover
    _HAS_CRITICAL = False
    CriticalState = None  # type: ignore[assignment]
    _roll_trajectory = None  # type: ignore[assignment]


try:  # pragma: no cover
    from .integrated_gradients import (
        flatten_observation as _ig_flatten_observation,
        observation_matrix as _ig_observation_matrix,
        policy_action_output as _ig_policy_action_output,
    )

    _HAS_IG_HELPERS = True
except Exception:  # pragma: no cover
    _HAS_IG_HELPERS = False
    _ig_flatten_observation = None  # type: ignore[assignment]
    _ig_observation_matrix = None  # type: ignore[assignment]
    _ig_policy_action_output = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_HIDDEN_SIZES: Tuple[int, ...] = (64, 64)
DEFAULT_LR = 3e-4
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 512
DEFAULT_L1_COEFF = 1e-2
DEFAULT_ENTROPY_COEFF = 0.0
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOPK_FRACTION = 0.1
DEFAULT_K = 1000
DEFAULT_SCORE_RANGE = (0.0, 1.0)

#: How the per-state importance signal is computed.
SCORE_MODES = ("hybrid", "attention", "sensitivity")
#: Feature-attention parameterisations.
ATTENTION_MODES = ("softmax", "sparsemax", "sigmoid")
#: Which policy output the attention module tries to preserve.
AIRS_TARGETS = ("action", "log_prob", "value")

#: Mask-net training sample budgets (Table 4) reused as the AIRS attention-module
#: training budget: Hopper/Walker2d/Reacher/HalfCheetah 3e5, Selfish Mining 1.5e6,
#: Cage Challenge 2 1e7, Autonomous Driving 2,443,260, Malware Mutation 32,349.
MASK_SAMPLE_BUDGETS: Dict[str, int] = {
    "hopper": 300_000,
    "walker2d": 300_000,
    "reacher": 300_000,
    "halfcheetah": 300_000,
    "selfish_mining": 1_500_000,
    "cage2": 10_000_000,
    "autodriving": 2_443_260,
    "malware_mutation": 32_349,
}


# ---------------------------------------------------------------------------
# Observation / policy-output helpers (delegate to IG when available)
# ---------------------------------------------------------------------------
def flatten_observation(observation: Any) -> np.ndarray:
    """Flatten an array/dict/tuple/scalar observation into a 1-D float32 vector."""
    if _HAS_IG_HELPERS:
        return _ig_flatten_observation(observation)
    if isinstance(observation, dict):
        parts = [np.asarray(observation[k], dtype=np.float32).reshape(-1) for k in sorted(observation.keys())]
        return np.concatenate(parts) if parts else np.zeros((0,), dtype=np.float32)
    if isinstance(observation, (tuple, list)) and len(observation) and not np.isscalar(observation[0]):
        return np.concatenate([np.asarray(o, dtype=np.float32).reshape(-1) for o in observation])
    return np.asarray(observation, dtype=np.float32).reshape(-1)


def observation_matrix(observations: Any) -> np.ndarray:
    """Normalise any observation container into an ``(N, D)`` float32 matrix."""
    if _HAS_IG_HELPERS:
        return _ig_observation_matrix(observations)
    if observations is None:
        return np.zeros((0, 0), dtype=np.float32)
    if isinstance(observations, np.ndarray) and observations.ndim == 2:
        return np.asarray(observations, dtype=np.float32)
    if isinstance(observations, np.ndarray) and observations.ndim == 1:
        return observations.reshape(1, -1).astype(np.float32)
    if hasattr(observations, "observations"):
        return np.asarray(observations.observations, dtype=np.float32).reshape(len(observations), -1)
    if isinstance(observations, (list, tuple)):
        if len(observations) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack([flatten_observation(o) for o in observations], axis=0)
    return flatten_observation(observations).reshape(1, -1)


def policy_action_output(policy: Any, obs_tensor: Any, target: str = "action", actions: Any = None) -> Any:
    """Differentiable policy output used as AIRS' action-preservation target."""
    if _HAS_IG_HELPERS:
        return _ig_policy_action_output(policy, obs_tensor, target=target, actions=actions)
    if not _HAS_TORCH or policy is None:
        return None
    try:
        if hasattr(policy, "get_distribution"):
            dist = policy.get_distribution(obs_tensor)
            if target in ("log_prob", "logprob"):
                return dist.log_prob(dist.mean)
            if hasattr(dist, "mean"):
                return dist.mean
        if hasattr(policy, "predict_values"):
            return policy.predict_values(obs_tensor)
        if callable(policy):
            return policy(obs_tensor)
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# AIRS attention network
# ---------------------------------------------------------------------------
if _HAS_TORCH:
    _AIRS_BASE: Tuple[type, ...] = (nn.Module,)
else:  # pragma: no cover
    _AIRS_BASE = (object,)


class AIRSNetwork(*_AIRS_BASE):  # type: ignore[misc]
    """Attention-based interpretable module producing AIRS explanations.

    The network learns to *preserve* the frozen policy's action output while
    attending to as few state features (and, for trajectories, as few steps) as
    possible.  The attention weights are the explanation.

    Args:
        obs_dim: Dimension of the (flattened) observation space.
        action_dim: Dimension of the action output being preserved.
        hidden_sizes: Hidden widths of the state encoder.
        activation: Activation function name.
        attention_mode: ``"softmax"`` (default), ``"sparsemax"`` or ``"sigmoid"``.
        temperature: Softmax temperature for the attention.
        temporal: Whether temporal (step) attention is also produced.
        feature_groups: Optional index groups (e.g. per-joint blocks) whose
            attention is pooled; ignored when ``None``.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
        activation: str = "tanh",
        attention_mode: str = "softmax",
        temperature: float = DEFAULT_TEMPERATURE,
        temporal: bool = True,
        feature_groups: Optional[Sequence[Sequence[int]]] = None,
    ) -> None:
        if not _HAS_TORCH:  # pragma: no cover
            raise ImportError("AIRSNetwork requires PyTorch; pip install torch")
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes) or DEFAULT_HIDDEN_SIZES
        self.activation_name = str(activation)
        self.attention_mode = str(attention_mode).lower()
        self.temperature = float(temperature)
        self.temporal = bool(temporal)
        self.feature_groups = (
            [tuple(int(i) for i in g) for g in feature_groups] if feature_groups else None
        )

        layers: List[Any] = []
        last = self.obs_dim
        for size in self.hidden_sizes:
            layers.append(nn.Linear(last, int(size)))
            layers.append(self._activation(self.activation_name))
            last = int(size)
        self.encoder = nn.Sequential(*layers)
        self.encoded_dim = last

        # Per-feature attention logits: one scalar score per state dimension.
        self.feature_attention = nn.Sequential(
            nn.Linear(self.encoded_dim, max(1, self.encoded_dim // 2)),
            self._activation(self.activation_name),
            nn.Linear(max(1, self.encoded_dim // 2), self.obs_dim),
        )
        # Temporal attention over steps (applied to the shared step embedding).
        self.temporal_attention = nn.Linear(self.encoded_dim, 1) if self.temporal else None
        # Action-preservation head reading the attention-masked state.
        self.action_head = nn.Sequential(
            nn.Linear(self.obs_dim, max(1, self.encoded_dim // 2)),
            self._activation(self.activation_name),
            nn.Linear(max(1, self.encoded_dim // 2), self.action_dim),
        )
        self._reset_parameters()

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _activation(name: str) -> Any:
        key = str(name).lower()
        if key == "relu":
            return nn.ReLU()
        if key in ("leaky_relu", "leakyrelu"):
            return nn.LeakyReLU()
        if key == "elu":
            return nn.ELU()
        if key == "gelu":
            return nn.GELU()
        return nn.Tanh()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Slightly random feature-attention init avoids an exactly-uniform start.
        with torch.no_grad():
            self.feature_attention[-1].weight.mul_(0.1)

    # -- attention -------------------------------------------------------
    def attention_logits(self, obs: Any) -> Any:
        embedding = self.encoder(obs)
        return self.feature_attention(embedding)

    def feature_weights(self, obs: Any) -> Any:
        """Per-state feature attention ``w(s) in Delta^d``."""
        logits = self.attention_logits(obs) / max(self.temperature, 1e-6)
        if self.attention_mode == "sigmoid":
            weights = torch.sigmoid(logits)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        elif self.attention_mode == "sparsemax":
            weights = _sparsemax(logits, dim=-1)
            denom = weights.sum(dim=-1, keepdim=True)
            uniform = torch.full_like(weights, 1.0 / max(self.obs_dim, 1))
            weights = torch.where(denom > 0, weights / denom.clamp_min(1e-8), uniform)
        else:
            weights = torch.softmax(logits, dim=-1)
        return weights

    def temporal_weights(self, embeddings: Any) -> Any:
        """Temporal (step) attention over a sequence ``(B, T, H)``."""
        if self.temporal_attention is None:
            batch, length = embeddings.shape[0], embeddings.shape[1]
            return torch.full(
                (batch, length), 1.0 / max(length, 1), device=embeddings.device, dtype=embeddings.dtype
            )
        logits = self.temporal_attention(embeddings).squeeze(-1)
        return torch.softmax(logits / max(self.temperature, 1e-6), dim=-1)

    # -- forward ---------------------------------------------------------
    def forward(self, obs: Any, return_dict: bool = True) -> Any:
        """Args: ``obs`` of shape ``(N, D)`` or ``(B, T, D)``.

        Returns a dict with ``feature_attention`` ``(..., D)``,
        ``temporal_attention`` ``(B, T)`` (or ``None``), ``masked_obs``, the
        predicted ``action`` and the shared ``embedding``.
        """
        if not _HAS_TORCH:  # pragma: no cover
            raise ImportError("AIRSNetwork requires PyTorch")
        obs = obs if torch.is_tensor(obs) else torch.as_tensor(np.asarray(obs, dtype=np.float32))
        obs = obs.float()
        if obs.dim() == 3:
            batch, length, _ = obs.shape
            flat = obs.reshape(batch * length, -1)
            embedding = self.encoder(flat).reshape(batch, length, -1)
            feature_attention = self.feature_weights(flat).reshape(batch, length, -1)
            temporal_attention = self.temporal_weights(embedding)
        else:
            flat = obs.reshape(-1, obs.shape[-1])
            embedding = self.encoder(flat)
            feature_attention = self.feature_weights(flat)
            temporal_attention = None

        masked_obs = flat * feature_attention
        action = self.action_head(masked_obs)
        result = {
            "feature_attention": feature_attention,
            "temporal_attention": temporal_attention,
            "masked_obs": masked_obs,
            "action": action,
            "embedding": embedding,
        }
        return result if return_dict else result["action"]

    def predict_action(self, obs: Any) -> Any:
        return self.forward(obs, return_dict=True)["action"]

    # -- loss ------------------------------------------------------------
    def airs_loss(
        self,
        obs: Any,
        target_action: Any,
        l1_coeff: float = DEFAULT_L1_COEFF,
        entropy_coeff: float = DEFAULT_ENTROPY_COEFF,
        temporal_sparsity: bool = True,
    ) -> Tuple[Any, Dict[str, float]]:
        """AIRS objective ``MSE(F(s), F_hat(s (*) w)) + l1_coeff * mean(w)``."""
        out = self.forward(obs, return_dict=True)
        predicted = out["action"]
        target = target_action if torch.is_tensor(target_action) else torch.as_tensor(
            np.asarray(target_action, dtype=np.float32)
        )
        target = target.float().reshape(predicted.shape)
        preservation = F.mse_loss(predicted, target)
        feature_attention = out["feature_attention"]
        sparsity = (
            feature_attention.mean()
            if feature_attention.numel()
            else torch.zeros((), device=predicted.device)
        )
        loss = preservation + float(l1_coeff) * sparsity

        temporal_entropy = torch.zeros((), device=predicted.device)
        if temporal_sparsity and out["temporal_attention"] is not None:
            temporal = out["temporal_attention"]
            log_temporal = torch.log(temporal.clamp_min(1e-8))
            temporal_entropy = -(temporal * log_temporal).sum(dim=-1).mean()
            loss = loss - float(entropy_coeff) * temporal_entropy
        stats = {
            "airs/preservation": float(preservation.detach().cpu().item()),
            "airs/sparsity": float(sparsity.detach().cpu().item()),
            "airs/temporal_entropy": float(temporal_entropy.detach().cpu().item()),
            "airs/loss": float(loss.detach().cpu().item()),
        }
        return loss, stats

    # -- persistence -----------------------------------------------------
    def save(self, path: str, extra: Optional[Dict[str, Any]] = None) -> str:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {
            "state_dict": self.state_dict(),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_sizes": tuple(self.hidden_sizes),
            "attention_mode": self.attention_mode,
            "temperature": self.temperature,
            "temporal": self.temporal,
            "anchor": "AIRS",
            "extra": dict(extra or {}),
        }
        torch.save(payload, path)
        return path

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "AIRSNetwork":
        payload = torch.load(path, map_location=device)
        network = cls(
            obs_dim=int(payload["obs_dim"]),
            action_dim=int(payload["action_dim"]),
            hidden_sizes=tuple(payload.get("hidden_sizes", DEFAULT_HIDDEN_SIZES)),
            attention_mode=payload.get("attention_mode", "softmax"),
            temperature=float(payload.get("temperature", DEFAULT_TEMPERATURE)),
            temporal=bool(payload.get("temporal", True)),
        )
        network.load_state_dict(payload["state_dict"], strict=False)
        network.to(device)
        return network

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"obs_dim={self.obs_dim}, action_dim={self.action_dim}, "
            f"hidden={self.hidden_sizes}, attention={self.attention_mode}"
        )


def _sparsemax(logits: Any, dim: int = -1) -> Any:
    """Sparsemax projection (Martins & Astudillo, 2016) used as an attention mode."""
    shift = logits.max(dim=dim, keepdim=True).values
    z = logits - shift
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    range_ = torch.arange(1, z.size(dim) + 1, device=z.device, dtype=z.dtype)
    shape = [1] * z.dim()
    shape[dim] = -1
    range_ = range_.view(shape)
    cumsum = z_sorted.cumsum(dim=dim)
    support = 1 + range_ * z_sorted > cumsum
    k = support.sum(dim=dim, keepdim=True).clamp_min(1)
    tau = cumsum.gather(dim, (k - 1).long()) / k.to(z.dtype)
    return torch.clamp(z - tau, min=0.0)


# ---------------------------------------------------------------------------
# Sensitivity / attention computation
# ---------------------------------------------------------------------------
def _local_policy_action(policy: Any, observation: Any, deterministic: bool = False) -> np.ndarray:
    """Query the frozen policy, normalising the many policy interfaces."""
    obs = np.asarray(observation, dtype=np.float32)
    single = obs.ndim == 1
    if single:
        obs = obs.reshape(1, -1)
    if policy is None:
        return np.zeros((obs.shape[0], 1), dtype=np.float32)
    try:
        if hasattr(policy, "predict"):
            action, _ = policy.predict(obs, deterministic=deterministic)
            return np.asarray(action, dtype=np.float32).reshape(obs.shape[0], -1)
        if hasattr(policy, "act"):
            action = policy.act(obs, deterministic=deterministic)
            return np.asarray(action, dtype=np.float32).reshape(obs.shape[0], -1)
        if callable(policy):
            action = policy(obs)
            return np.asarray(action, dtype=np.float32).reshape(obs.shape[0], -1)
    except TypeError:
        try:
            action = policy.act(obs) if hasattr(policy, "act") else policy(obs)
            return np.asarray(action, dtype=np.float32).reshape(obs.shape[0], -1)
        except Exception:
            pass
    except Exception:
        pass
    return np.zeros((obs.shape[0], 1), dtype=np.float32)


def _policy_sensitivity_fd(
    policy: Any,
    observations: np.ndarray,
    eps: float = 1e-3,
) -> Optional[np.ndarray]:
    """Finite-difference per-dimension sensitivity ``|dF/ds_j|`` of the policy.

    Returns an ``(N, D)`` matrix or ``None`` when the policy cannot be queried.
    A per-dimension scale keeps the perturbation meaningful across state units.
    """
    if policy is None or observations.size == 0:
        return None
    obs = np.asarray(observations, dtype=np.float32)
    n, dim = obs.shape
    scale = np.maximum(np.std(obs, axis=0), 1e-3).astype(np.float32)
    step = float(eps) * scale
    try:
        base = _local_policy_action(policy, obs, deterministic=True).reshape(n, -1)
    except Exception:
        return None
    if base.size == 0:
        return None
    sensitivity = np.zeros((n, dim), dtype=np.float32)
    for j in range(dim):
        perturb = np.array(obs, copy=True)
        perturb[:, j] = perturb[:, j] + step[j]
        try:
            shifted = _local_policy_action(policy, perturb, deterministic=True).reshape(n, -1)
        except Exception:
            return None
        diff = np.linalg.norm(shifted - base, axis=1) / max(float(step[j]), 1e-8)
        sensitivity[:, j] = np.abs(diff).astype(np.float32)
    return sensitivity


def _attention_weights_numpy(
    network: Optional[Any],
    observations: np.ndarray,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Optional[np.ndarray]:
    """Evaluate the AIRS feature attention in numpy, batched."""
    if network is None or not _HAS_TORCH:
        return None
    obs = np.asarray(observations, dtype=np.float32)
    if obs.size == 0:
        return None
    chunks: List[np.ndarray] = []
    network.eval()
    with torch.no_grad():
        chunk_size = max(1, int(batch_size))
        for start in range(0, obs.shape[0], chunk_size):
            chunk = obs[start : start + chunk_size]
            tensor = torch.as_tensor(chunk, dtype=torch.float32)
            out = network.forward(tensor, return_dict=True)
            chunks.append(out["feature_attention"].detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0) if chunks else None


def _hashed_scores(observations: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    """Deterministic pseudo-random scores used as the last-resort fallback."""
    obs = np.asarray(observations, dtype=np.float32)
    if obs.size == 0:
        return np.zeros((0,), dtype=np.float32)
    base_seed = 0 if seed is None else int(seed)
    out = np.empty((obs.shape[0],), dtype=np.float32)
    for i in range(obs.shape[0]):
        digest = hash((base_seed, round(float(np.sum(np.abs(obs[i]))), 6), i)) % (2 ** 31)
        out[i] = digest / float(2 ** 31)
    return out


def _normalize_scores(scores: np.ndarray, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if scores.size == 0:
        return scores
    finite = np.isfinite(scores)
    if not np.any(finite):
        return np.full_like(scores, float((low + high) / 2.0))
    lo = float(np.nanmin(scores[finite]))
    hi = float(np.nanmax(scores[finite]))
    if hi - lo < 1e-12:
        return np.full_like(scores, float((low + high) / 2.0))
    scaled = (scores - lo) / (hi - lo)
    return (low + scaled * (high - low)).astype(np.float32)


def _combine_scores(
    attention: Optional[np.ndarray],
    sensitivity: Optional[np.ndarray],
    mode: str = "hybrid",
) -> np.ndarray:
    """Combine feature attention and sensitivity into per-state importance."""
    mode = (mode or "hybrid").lower()
    if mode not in SCORE_MODES:
        mode = "hybrid"
    if attention is not None and sensitivity is not None:
        if mode == "attention":
            scores = attention.sum(axis=1)
        elif mode == "sensitivity":
            scores = sensitivity.sum(axis=1)
        else:  # hybrid: attention-weighted sensitivity
            scores = np.sum(np.asarray(attention) * np.asarray(sensitivity), axis=1)
            if not np.any(np.isfinite(scores)) or float(np.nanmax(scores)) <= 1e-12:
                scores = sensitivity.sum(axis=1)
    elif sensitivity is not None and mode != "attention":
        scores = sensitivity.sum(axis=1)
    elif attention is not None:
        scores = attention.sum(axis=1)
    else:
        scores = np.zeros((0,), dtype=np.float32)
    return np.asarray(scores, dtype=np.float32).reshape(-1)


# ---------------------------------------------------------------------------
# Local fallbacks for the importance containers
# ---------------------------------------------------------------------------
def _local_aggregate_importance(scores: np.ndarray, mode: str = "mean", topk_frac: float = DEFAULT_TOPK_FRACTION) -> float:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        return float("nan")
    mode = (mode or "mean").lower()
    if mode == "max":
        return float(np.max(scores))
    if mode == "sum":
        return float(np.sum(scores))
    if mode == "last":
        return float(scores[-1])
    if mode == "first":
        return float(scores[0])
    if mode == "topk_mean":
        k = min(max(1, int(round(topk_frac * scores.size))), scores.size)
        return float(np.mean(np.sort(scores)[-k:]))
    return float(np.mean(scores))


def _local_argmax_importance(scores: np.ndarray) -> int:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.size == 0 or np.all(np.isnan(scores)):
        return 0
    return int(np.nanargmax(scores))


def _local_rank_states(scores: np.ndarray, descending: bool = True) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        return np.zeros((0,), dtype=np.int64)
    order = np.argsort(-scores, kind="mergesort") if descending else np.argsort(scores, kind="mergesort")
    return order.astype(np.int64)


def _local_top_k_indices(scores: np.ndarray, k: int = 10) -> np.ndarray:
    return _local_rank_states(scores, descending=True)[: max(0, int(k))]


def _local_extract_observations(trajectory: Any) -> np.ndarray:
    if trajectory is None:
        return np.zeros((0,), dtype=np.float32)
    for attr in ("observations", "obs"):
        value = getattr(trajectory, attr, None)
        if value is not None:
            return np.asarray(value, dtype=np.float32)
    if isinstance(trajectory, (list, tuple)):
        if len(trajectory) == 0:
            return np.zeros((0,), dtype=np.float32)
        first = trajectory[0]
        if isinstance(first, (list, tuple)) and len(first) and np.isscalar(first[0]):
            return np.asarray(trajectory, dtype=np.float32)
        obs = []
        for item in trajectory:
            if hasattr(item, "observation"):
                obs.append(np.asarray(item.observation, dtype=np.float32).reshape(-1))
            elif isinstance(item, (list, tuple)) and len(item) and not np.isscalar(item[0]):
                obs.append(np.asarray(item[0], dtype=np.float32).reshape(-1))
            else:
                obs.append(np.asarray(item, dtype=np.float32).reshape(-1))
        try:
            return np.stack(obs, axis=0)
        except Exception:
            return np.asarray(obs, dtype=np.float32)
    return np.asarray(trajectory, dtype=np.float32)


def _extract_observations_any(trajectory: Any) -> np.ndarray:
    if _HAS_IMPORTANCE and _extract_observations is not None:
        try:
            return np.asarray(_extract_observations(trajectory), dtype=np.float32)
        except Exception:
            pass
    return _local_extract_observations(trajectory)


class _FallbackTrajectoryImportance:
    """Minimal stand-in for ``rice.explanation.importance.TrajectoryImportance``."""

    def __init__(
        self,
        scores: np.ndarray,
        observations: Optional[np.ndarray] = None,
        indices: Optional[np.ndarray] = None,
        mode: str = "mean",
        aggregate: Optional[float] = None,
        critical_index: Optional[int] = None,
    ) -> None:
        self.scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        self.observations = observations
        self.indices = indices if indices is not None else np.arange(self.scores.size)
        self.mode = mode
        self.aggregate = (
            float(aggregate) if aggregate is not None else _local_aggregate_importance(self.scores, mode)
        )
        self.critical_index = (
            int(critical_index) if critical_index is not None else _local_argmax_importance(self.scores)
        )

    def __len__(self) -> int:
        return int(self.scores.size)

    def top_k(self, k: int = 10) -> np.ndarray:
        return _local_top_k_indices(self.scores, k)

    def to_dict(self, include_scores: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "mode": self.mode,
            "aggregate": self.aggregate,
            "critical_index": self.critical_index,
            "length": int(self.scores.size),
        }
        if include_scores:
            payload["scores"] = self.scores.tolist()
        return payload


class _FallbackCriticalState:
    """Duck-typed stand-in for ``rice.explanation.critical_state.CriticalState``."""

    def __init__(
        self,
        index: int,
        observation: Any,
        state: Any = None,
        actions: Optional[List[Any]] = None,
        score: float = float("nan"),
        importance_scores: Optional[np.ndarray] = None,
        trajectory_length: int = 0,
        env_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.index = int(index)
        self.observation = observation
        self.state = state
        self.actions = list(actions) if actions else []
        self.score = float(score)
        self.importance_scores = importance_scores
        self.trajectory_length = int(trajectory_length)
        self.env_id = env_id
        self.metadata = dict(metadata or {})

    @property
    def restore_payload(self) -> Any:
        if self.state is None:
            return None
        if isinstance(self.state, dict) and "state" in self.state and "kind" in self.state:
            return self.state
        return {"kind": "sim", "state": self.state}

    def to_dict(self, include_scores: bool = False) -> Dict[str, Any]:
        payload = {
            "index": self.index,
            "score": self.score,
            "trajectory_length": self.trajectory_length,
            "env_id": self.env_id,
            "metadata": dict(self.metadata),
        }
        if include_scores and self.importance_scores is not None:
            payload["importance_scores"] = np.asarray(self.importance_scores).tolist()
        return payload


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class AIRSConfig:
    """Configuration for the AIRS explanation baseline (Experiment III / Table 6)."""

    env_id: str = "default"
    seed: Optional[int] = None
    # attention module
    hidden_sizes: Tuple[int, ...] = DEFAULT_HIDDEN_SIZES
    activation: str = "tanh"
    attention_mode: str = "softmax"
    temperature: float = DEFAULT_TEMPERATURE
    score_mode: str = "hybrid"
    temporal: bool = True
    # training the interpretable attention module
    lr: float = DEFAULT_LR
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    l1_coeff: float = DEFAULT_L1_COEFF
    entropy_coeff: float = DEFAULT_ENTROPY_COEFF
    train_samples: int = 20000
    target: str = "action"
    # scoring
    sensitivity_eps: float = 1e-3
    normalize: bool = True
    score_range: Tuple[float, float] = DEFAULT_SCORE_RANGE
    topk_fraction: float = DEFAULT_TOPK_FRACTION
    deterministic_policy: bool = True
    attach_scores: bool = True
    with_mask_shim: bool = False
    device: str = "cpu"
    K: Optional[int] = None

    def __post_init__(self) -> None:
        self.env_id = _normalize_env_key(self.env_id)
        self.hidden_sizes = tuple(int(h) for h in self.hidden_sizes)
        self.score_mode = self.score_mode if self.score_mode in SCORE_MODES else "hybrid"
        self.attention_mode = (
            self.attention_mode if self.attention_mode in ATTENTION_MODES else "softmax"
        )
        if self.target not in AIRS_TARGETS:
            self.target = "action"
        self.score_range = (float(self.score_range[0]), float(self.score_range[1]))

    @classmethod
    def from_dict(cls, cfg: Optional[Any] = None, **overrides: Any) -> "AIRSConfig":
        """Build a config from a (possibly nested) dict, tolerating aliases."""
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        alias = {
            "env": "env_id",
            "env_key": "env_id",
            "application": "env_id",
            "hidden": "hidden_sizes",
            "net_arch": "hidden_sizes",
            "layers": "hidden_sizes",
            "learning_rate": "lr",
            "n_epochs": "epochs",
            "num_epochs": "epochs",
            "bs": "batch_size",
            "batch": "batch_size",
            "l1": "l1_coeff",
            "lambda_l1": "l1_coeff",
            "sparsity": "l1_coeff",
            "entropy": "entropy_coeff",
            "eps": "sensitivity_eps",
            "m": "n_steps_removed",
            "steps": "n_steps_removed",
            "topk_frac": "topk_fraction",
            "topk": "topk_fraction",
        }
        data: Dict[str, Any] = {}
        if cfg is not None:
            if isinstance(cfg, AIRSConfig):
                data.update(cfg.to_dict())
            elif isinstance(cfg, dict):
                flat: Dict[str, Any] = {}
                for section in ("airs", "AIRS", "explanation", "baseline", "default"):
                    sub = cfg.get(section)
                    if isinstance(sub, dict):
                        flat.update(sub)
                flat.update({k: v for k, v in cfg.items() if not isinstance(v, dict)})
                data.update(flat)
            elif hasattr(cfg, "items"):
                data.update(dict(cfg.items()))
        data.update(overrides)
        cleaned: Dict[str, Any] = {}
        for key, value in data.items():
            mapped = alias.get(key, key)
            if mapped and mapped in known:
                cleaned[mapped] = value
        return cls(**cleaned)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["hidden_sizes"] = list(self.hidden_sizes)
        payload["score_range"] = list(self.score_range)
        return payload


# ---------------------------------------------------------------------------
# Functional scoring API
# ---------------------------------------------------------------------------
def airs_importance_scores(
    policy: Any,
    observations: Any,
    network: Optional[Any] = None,
    target: str = "action",
    score_mode: str = "hybrid",
    normalize: bool = True,
    score_range: Tuple[float, float] = DEFAULT_SCORE_RANGE,
    sensitivity_eps: float = 1e-3,
    batch_size: int = DEFAULT_BATCH_SIZE,
    rng: Optional[Any] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Step-level AIRS importance scores for a batch of observations.

    The score is the attention-weighted per-dimension sensitivity of the frozen
    policy's output, ``sum_j w_j(s) * |dF(s)/ds_j|``, optionally min-max
    normalised into ``[0, 1]`` (comparable to the mask network's ``P(keep)``).

    Fallback chain: attention-weighted sensitivity -> plain sensitivity ->
    deterministic hashed scores.
    """
    obs = observation_matrix(observations)
    if obs.size == 0:
        return np.zeros((0,), dtype=np.float32)

    if rng is None:
        rng = _get_rng(seed)

    attention = _attention_weights_numpy(network, obs, batch_size=batch_size)
    sensitivity = _policy_sensitivity_fd(policy, obs, eps=sensitivity_eps)
    scores = _combine_scores(attention, sensitivity, mode=score_mode)
    if scores.size != obs.shape[0]:
        scores = _hashed_scores(obs, seed=seed)
    if normalize:
        scores = _normalize_scores(scores, low=score_range[0], high=score_range[1])
    return np.asarray(scores, dtype=np.float32).reshape(-1)


#: Alias used by experiment drivers / the baseline registry.
airs_scores = airs_importance_scores


def train_airs_network(
    network: Any,
    policy: Any,
    observations: Any,
    target: str = "action",
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    l1_coeff: float = DEFAULT_L1_COEFF,
    entropy_coeff: float = DEFAULT_ENTROPY_COEFF,
    device: str = "cpu",
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Any:
    """Train the AIRS attention module to preserve the frozen policy's action.

    Objective (attention-weighted action preservation + attention sparsity)::

        L = || F(s) - F_hat(s (*) w(s)) ||^2 + l1_coeff * mean(w(s))

    Args:
        network: the :class:`AIRSNetwork` to train (in place).
        policy: the frozen target policy pi whose action output must be preserved.
        observations: ``(N, D)`` states sampled from pi's visitation distribution.
        target: which policy output to preserve (``"action"``/``"log_prob"``/``"value"``).

    Returns:
        The (mutated) network.
    """
    if not _HAS_TORCH or network is None:  # pragma: no cover
        return network
    obs = observation_matrix(observations)
    if obs.size == 0:
        return network
    if seed is not None:
        torch.manual_seed(seed)

    tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
    with torch.no_grad():
        target_tensor = policy_action_output(policy, tensor, target=target)
        if target_tensor is None:
            return network
        target_tensor = torch.as_tensor(target_tensor, dtype=torch.float32, device=device)

    network.to(device)
    network.train()
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    n = tensor.shape[0]
    chunk = max(1, int(batch_size))
    for epoch in range(int(epochs)):
        permutation = torch.randperm(n, device=device)
        for start in range(0, n, chunk):
            idx = permutation[start : start + chunk]
            loss, _stats = network.airs_loss(
                tensor[idx],
                target_tensor[idx],
                l1_coeff=l1_coeff,
                entropy_coeff=entropy_coeff,
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 0.5)
            optimizer.step()
        if verbose:  # pragma: no cover - diagnostics only
            _get_logger("rice.baselines.airs").info(
                "AIRS epoch %d/%d loss=%.6f", epoch + 1, epochs, float(loss.detach().cpu().item())
            )
    network.eval()
    return network


def collect_policy_observations(
    env: Any,
    policy: Any,
    n_samples: int = 20000,
    seed: Optional[int] = None,
    deterministic: bool = True,
    max_steps: Optional[int] = None,
) -> np.ndarray:
    """Roll the frozen policy to gather training states for the AIRS module."""
    rng = _get_rng(seed)
    collected: List[np.ndarray] = []
    total = 0
    horizon = int(max_steps or DEFAULT_K)
    while total < int(n_samples):
        try:
            result = env.reset(seed=int(rng.randint(0, 2 ** 31 - 1)))
        except TypeError:
            result = env.reset()
        obs = result[0] if isinstance(result, (tuple, list)) and result else result
        for _ in range(horizon):
            obs_arr = flatten_observation(obs)
            collected.append(obs_arr)
            total += 1
            action = _local_policy_action(policy, obs_arr, deterministic=deterministic)[0]
            try:
                result = env.step(action)
            except Exception:
                break
            if not isinstance(result, (tuple, list)):
                break
            if len(result) == 5:
                obs, _reward, terminated, truncated, _info = result
                done = bool(terminated or truncated)
            else:
                obs, _reward, done, _info = result
            if done or total >= int(n_samples):
                break
    if not collected:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack(collected, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Local trajectory rollout fallback
# ---------------------------------------------------------------------------
class _LocalRollout:
    """Minimal trajectory container used when ``critical_state`` is unavailable."""

    def __init__(
        self,
        observations: np.ndarray,
        actions: Optional[np.ndarray] = None,
        rewards: Optional[np.ndarray] = None,
        next_observations: Optional[np.ndarray] = None,
        dones: Optional[np.ndarray] = None,
        infos: Optional[List[Dict[str, Any]]] = None,
        states: Optional[List[Any]] = None,
        env_id: str = "default",
    ) -> None:
        self.observations = np.asarray(observations, dtype=np.float32)
        self.actions = actions if actions is not None else np.zeros((len(self.observations), 1), dtype=np.float32)
        self.rewards = rewards if rewards is not None else np.zeros((len(self.observations),), dtype=np.float32)
        self.next_observations = next_observations
        self.dones = dones if dones is not None else np.zeros((len(self.observations),), dtype=bool)
        self.infos = infos if infos is not None else [{} for _ in range(len(self.observations))]
        self.states = states if states is not None else [None] * len(self.observations)
        self.env_id = env_id
        self.length = int(len(self.observations))
        self.terminated_early = bool(np.any(self.dones)) if self.dones.size else False

    def __len__(self) -> int:
        return self.length

    @property
    def importance_observations(self) -> np.ndarray:
        return self.observations

    def to_dict(self, include_arrays: bool = False) -> Dict[str, Any]:
        payload = {"env_id": self.env_id, "length": self.length, "terminated_early": self.terminated_early}
        if include_arrays:
            payload["observations"] = np.asarray(self.observations).tolist()
            payload["actions"] = np.asarray(self.actions).tolist()
            payload["rewards"] = np.asarray(self.rewards).tolist()
        return payload


def _local_get_state(env: Any) -> Any:
    try:  # pragma: no cover - depends on the backend
        from ..envs.reset_wrapper import get_env_state

        return get_env_state(env)
    except Exception:
        pass
    value = getattr(env, "state", None)
    if value is not None:
        try:
            return value() if callable(value) else value
        except Exception:
            pass
    inner = getattr(env, "env", None)
    if inner is not None and inner is not env:
        return _local_get_state(inner)
    return None


def _local_roll_trajectory(
    env: Any,
    policy: Any,
    length: Optional[int] = None,
    reset: bool = True,
    deterministic: bool = False,
    seed: Optional[int] = None,
    stop_on_done: bool = True,
    env_id: str = "default",
) -> _LocalRollout:
    if length is None:
        length = DEFAULT_K
        for attr in ("rice_max_episode_steps",):
            value = getattr(env, attr, None)
            if value:
                length = int(value)
                break
    length = int(length)
    observations, actions, rewards, next_observations, dones, infos, states = [], [], [], [], [], [], []
    obs = None
    if reset:
        try:
            result = env.reset(seed=seed) if seed is not None else env.reset()
        except TypeError:
            result = env.reset()
        obs = result[0] if isinstance(result, (tuple, list)) and result else result
    for _ in range(max(0, length)):
        if obs is None:
            break
        obs_arr = flatten_observation(obs)
        action = _local_policy_action(policy, obs_arr, deterministic=deterministic)[0]
        try:
            step_result = env.step(action)
        except Exception:
            break
        if not isinstance(step_result, (tuple, list)):
            break
        if len(step_result) == 5:
            next_obs, reward, terminated, truncated, info = step_result
            done = bool(terminated or truncated)
        else:
            next_obs, reward, done, info = step_result
        observations.append(obs_arr)
        actions.append(np.asarray(action).reshape(-1))
        rewards.append(float(reward))
        next_observations.append(flatten_observation(next_obs))
        dones.append(bool(done))
        infos.append(dict(info) if isinstance(info, dict) else {})
        states.append(_local_get_state(env))
        obs = next_obs
        if done and stop_on_done:
            break
    if not observations:
        return _LocalRollout(np.zeros((0, 0), dtype=np.float32), env_id=env_id)
    return _LocalRollout(
        np.stack(observations, axis=0),
        np.stack(actions, axis=0),
        np.asarray(rewards, dtype=np.float32),
        np.stack(next_observations, axis=0),
        np.asarray(dones, dtype=bool),
        infos,
        states,
        env_id=env_id,
    )


# ---------------------------------------------------------------------------
# Scorer (drop-in for rice.explanation.importance.ImportanceScorer)
# ---------------------------------------------------------------------------
class AIRSScorer:
    """Step-level AIRS scorer mirroring ``ImportanceScorer``'s surface."""

    is_random = False
    name = "AIRS"

    def __init__(
        self,
        policy: Any = None,
        network: Optional[Any] = None,
        config: Optional[Any] = None,
        env_id: str = "default",
        rng: Optional[Any] = None,
        seed: Optional[int] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        score_mode: Optional[str] = None,
        normalize: Optional[bool] = None,
        mask_net: Any = None,
        **kwargs: Any,
    ) -> None:
        self.config = config if isinstance(config, AIRSConfig) else AIRSConfig.from_dict(config or {})
        if env_id and env_id != "default":
            self.config.env_id = _normalize_env_key(env_id)
        if score_mode is not None:
            self.config.score_mode = score_mode
        if normalize is not None:
            self.config.normalize = bool(normalize)
        if batch_size is not None:
            self.config.batch_size = int(batch_size)
        if device is not None:
            self.config.device = str(device)
        self.policy = policy
        self.network = network
        self.mask_net = mask_net  # kept for API parity; unused by AIRS
        self.env_id = self.config.env_id
        self.rng = rng if rng is not None else _get_rng(seed if seed is not None else self.config.seed)
        self.calls = 0
        self.total_states = 0
        self._logger = _get_logger("rice.baselines.airs")

    # -- core ------------------------------------------------------------
    def score(self, observations: Any) -> np.ndarray:
        """Return per-state AIRS importance scores (higher = more critical)."""
        scores = airs_importance_scores(
            self.policy,
            observations,
            network=self.network,
            score_mode=self.config.score_mode,
            normalize=self.config.normalize,
            score_range=self.config.score_range,
            sensitivity_eps=self.config.sensitivity_eps,
            batch_size=self.config.batch_size,
            rng=self.rng,
            seed=self.config.seed,
        )
        self.calls += 1
        self.total_states += int(scores.size)
        return scores

    # aliases expected by the drivers
    def score_observations(self, observations: Any, batch_size: Optional[int] = None) -> np.ndarray:
        return self.score(observations)

    def score_batch(self, observations: Any, batch_size: Optional[int] = None) -> np.ndarray:
        return self.score(observations)

    def __call__(self, observations: Any) -> np.ndarray:
        return self.score(observations)

    def score_trajectory(self, trajectory: Any) -> Any:
        observations = _extract_observations_any(trajectory)
        scores = self.score(observations)
        aggregate = _local_aggregate_importance(scores, "mean")
        critical_index = _local_argmax_importance(scores)
        if _HAS_IMPORTANCE and TrajectoryImportance is not None:
            try:
                return TrajectoryImportance(
                    scores=scores,
                    observations=np.asarray(observations, dtype=np.float32),
                    indices=np.arange(scores.size),
                    mode="mean",
                    aggregate=aggregate,
                    critical_index=critical_index,
                )
            except Exception:
                pass
        return _FallbackTrajectoryImportance(
            scores,
            observations=np.asarray(observations, dtype=np.float32),
            mode="mean",
            aggregate=aggregate,
            critical_index=critical_index,
        )

    def importance_scores(self, trajectory: Any) -> np.ndarray:
        return np.asarray(self.score_trajectory(trajectory).scores, dtype=np.float32)

    def most_important_index(self, trajectory: Any) -> int:
        return int(np.asarray(self.score_trajectory(trajectory).critical_index))

    def most_important_state(self, trajectory: Any) -> Tuple[int, np.ndarray]:
        observations = np.asarray(_extract_observations_any(trajectory), dtype=np.float32)
        idx = self.most_important_index(trajectory)
        if observations.size == 0:
            return 0, np.zeros((0,), dtype=np.float32)
        return idx, np.asarray(observations[idx], dtype=np.float32).reshape(-1)

    def top_k_states(self, trajectory: Any, k: int = 10) -> List[Tuple[int, np.ndarray]]:
        observations = np.asarray(_extract_observations_any(trajectory), dtype=np.float32)
        scores = self.score(observations)
        order = _local_top_k_indices(scores, k)
        out: List[Tuple[int, np.ndarray]] = []
        for idx in order:
            if observations.size == 0:
                break
            out.append((int(idx), np.asarray(observations[int(idx)], dtype=np.float32).reshape(-1)))
        return out

    def rank(self, trajectory: Any) -> np.ndarray:
        observations = _extract_observations_any(trajectory)
        return _local_rank_states(self.score(observations), descending=True)

    def ranked_observations(self, trajectory: Any) -> np.ndarray:
        observations = np.asarray(_extract_observations_any(trajectory), dtype=np.float32)
        order = self.rank(trajectory)
        if observations.size == 0:
            return observations
        return observations[order]

    def trajectory_aggregate(self, trajectory: Any, mode: Optional[str] = None) -> float:
        observations = _extract_observations_any(trajectory)
        return _local_aggregate_importance(self.score(observations), mode or "mean")

    def summary(self, trajectory: Any) -> Dict[str, Any]:
        observations = _extract_observations_any(trajectory)
        scores = self.score(observations)
        summary = {
            "n": float(scores.size),
            "mean": float(np.mean(scores)) if scores.size else float("nan"),
            "max": float(np.max(scores)) if scores.size else float("nan"),
            "argmax": float(_local_argmax_importance(scores)),
            "mode": self.config.score_mode,
            "explanation": self.name,
        }
        return summary

    def attach_to_wrapper(self, wrapper: Any, trajectory: Any = None) -> Any:
        if not self.config.attach_scores:
            return wrapper
        if trajectory is None:
            return wrapper
        observations = _extract_observations_any(trajectory)
        scores = self.score(observations)
        try:
            setattr(wrapper, "last_importance_scores", scores)
            snapshots = getattr(wrapper, "snapshots", None)
            if snapshots is not None and len(snapshots) >= scores.size:
                for snap, value in zip(snapshots[-scores.size :], scores):
                    try:
                        snap.score = float(value)
                    except Exception:
                        continue
        except Exception:
            pass
        return wrapper

    def keep_probability(self, observations: Any) -> np.ndarray:
        return self.score(observations)

    def blind_probability(self, observations: Any) -> np.ndarray:
        return 1.0 - self.score(observations)

    @property
    def stats(self) -> Dict[str, Any]:
        mean = float(self.total_states / self.calls) if self.calls else 0.0
        return {
            "scorer": "AIRS",
            "calls": self.calls,
            "total_states": self.total_states,
            "mean_states_per_call": mean,
            "score_mode": self.config.score_mode,
            "has_attention_network": bool(self.network is not None and _HAS_TORCH),
        }

    def reset_statistics(self) -> None:
        self.calls = 0
        self.total_states = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"config": self.config.to_dict(), "stats": self.stats, "name": self.name}


class _AIRSMaskShim:
    """Mask-network-compatible shim backed by AIRS attention/sensitivity scores."""

    def __init__(self, scorer: AIRSScorer, env_key: str = "default") -> None:
        self.scorer = scorer
        self.env_key = env_key

    def keep_probability(self, observations: Any) -> np.ndarray:
        return self.scorer.score(observations)

    def blind_probability(self, observations: Any) -> np.ndarray:
        return 1.0 - self.scorer.score(observations)

    def score(self, observations: Any) -> np.ndarray:
        return self.scorer.score(observations)

    def state_importance(self, observations: Any, batch_size: int = DEFAULT_BATCH_SIZE) -> np.ndarray:
        return self.scorer.score(observations)

    def __call__(self, observations: Any) -> np.ndarray:
        return self.scorer.score(observations)


# ---------------------------------------------------------------------------
# Critical-state selector (drop-in for CriticalStateSelector)
# ---------------------------------------------------------------------------
class AIRSStateSelector:
    """Selects the AIRS-most-important visited state as the critical state."""

    is_random = False
    name = "AIRS"

    def __init__(
        self,
        env: Any = None,
        policy: Any = None,
        mask_net: Any = None,
        K: Optional[int] = None,
        scorer: Optional[AIRSScorer] = None,
        deterministic_policy: bool = True,
        batch_size: Optional[int] = None,
        device: Optional[str] = None,
        rng: Optional[Any] = None,
        seed: Optional[int] = None,
        attach_scores: bool = True,
        cache: bool = False,
        config: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        self.config = config if isinstance(config, AIRSConfig) else AIRSConfig.from_dict(config or {})
        self.env = env
        self.policy = policy
        self.mask_net = mask_net  # API parity
        self.K = K
        self.rng = rng if rng is not None else _get_rng(seed if seed is not None else self.config.seed)
        self.scorer = scorer or AIRSScorer(
            policy=policy,
            config=self.config,
            env_id=self.config.env_id,
            rng=self.rng,
            seed=self.config.seed,
            batch_size=batch_size,
            device=device,
        )
        self.attach_scores = bool(attach_scores)
        self.cache = bool(cache)
        self.deterministic_policy = bool(deterministic_policy)
        self.env_id = self.config.env_id
        self.history: List[Any] = []
        self.stats: Dict[str, Any] = {"rollouts": 0, "mean_length": 0.0, "mean_critical_index": 0.0}
        self.last_critical_state: Any = None
        self.last_rollout: Any = None

    # -- scoring pass-through -------------------------------------------
    def score(self, observations: Any) -> np.ndarray:
        return self.scorer.score(observations)

    def score_trajectory(self, trajectory: Any) -> Any:
        return self.scorer.score_trajectory(trajectory)

    def importance_scores(self, trajectory: Any) -> np.ndarray:
        return self.scorer.importance_scores(trajectory)

    # -- rollout / selection --------------------------------------------
    def rollout(
        self,
        policy: Any = None,
        K: Optional[int] = None,
        reset: bool = True,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        target_policy = policy if policy is not None else self.policy
        length = K if K is not None else self.K
        if _HAS_CRITICAL and _roll_trajectory is not None:
            try:
                return _roll_trajectory(
                    self.env,
                    target_policy,
                    length=length,
                    reset=reset,
                    deterministic=self.deterministic_policy,
                    seed=seed,
                    collect_states=True,
                )
            except Exception:
                pass
        return _local_roll_trajectory(
            self.env,
            target_policy,
            length=length,
            reset=reset,
            deterministic=self.deterministic_policy,
            seed=seed,
            env_id=self.env_id,
        )

    def _build_critical_state(self, rollout: Any, index: int, scores: np.ndarray) -> Any:
        observations = np.asarray(getattr(rollout, "observations", np.zeros((0, 0))), dtype=np.float32)
        if observations.size == 0:
            observations = _local_extract_observations(rollout)
        if observations.size:
            index = int(np.clip(index, 0, observations.shape[0] - 1))
            observation = np.asarray(observations[index], dtype=np.float32).reshape(-1)
        else:
            index = 0
            observation = np.zeros((0,), dtype=np.float32)
        states = getattr(rollout, "states", None)
        state = states[index] if states is not None and index < len(states) else None
        actions = getattr(rollout, "actions", None)
        prior_actions = (
            [np.asarray(a).reshape(-1) for a in actions[:index]]
            if actions is not None and len(actions) > index
            else []
        )
        score = float(scores[index]) if scores.size > index else float("nan")
        length = int(getattr(rollout, "length", observations.shape[0] if observations.size else 0))
        metadata = {
            "explanation": "AIRS",
            "score_mode": self.config.score_mode,
            "engine": "attention" if self.scorer.network is not None else "sensitivity",
        }
        if _HAS_CRITICAL and CriticalState is not None:
            try:
                return CriticalState(
                    index=int(index),
                    observation=observation,
                    state=state,
                    actions=prior_actions,
                    score=score,
                    importance_scores=scores,
                    trajectory_length=length,
                    env_id=self.env_id,
                    metadata=metadata,
                )
            except Exception:
                pass
        return _FallbackCriticalState(
            index=int(index),
            observation=observation,
            state=state,
            actions=prior_actions,
            score=score,
            importance_scores=scores,
            trajectory_length=length,
            env_id=self.env_id,
            metadata=metadata,
        )

    def select(
        self,
        policy: Any = None,
        K: Optional[int] = None,
        reset: bool = True,
        seed: Optional[int] = None,
        rollout: Any = None,
        **kwargs: Any,
    ) -> Any:
        if rollout is None:
            rollout = self.rollout(policy=policy, K=K, reset=reset, seed=seed)
        observations = np.asarray(getattr(rollout, "observations", np.zeros((0, 0))), dtype=np.float32)
        scores = self.score(observations)
        index = _local_argmax_importance(scores)
        critical = self._build_critical_state(rollout, index, scores)
        if self.attach_scores and self.env is not None:
            try:
                self.scorer.attach_to_wrapper(self.env, rollout)
            except Exception:
                pass
        if self.cache:
            self.last_critical_state = critical
            self.last_rollout = rollout
        self.history.append(critical)
        self._update_stats(
            int(index), int(getattr(rollout, "length", observations.shape[0]) if observations.size else 0)
        )
        return critical

    def select_top_k(
        self,
        k: int = 10,
        policy: Any = None,
        K: Optional[int] = None,
        reset: bool = True,
        seed: Optional[int] = None,
        rollout: Any = None,
        **kwargs: Any,
    ) -> List[Any]:
        if rollout is None:
            rollout = self.rollout(policy=policy, K=K, reset=reset, seed=seed)
        observations = np.asarray(getattr(rollout, "observations", np.zeros((0, 0))), dtype=np.float32)
        scores = self.score(observations)
        order = _local_top_k_indices(scores, k)
        return [self._build_critical_state(rollout, int(i), scores) for i in order]

    def _update_stats(self, index: int, length: int) -> None:
        rollouts = int(self.stats.get("rollouts", 0)) + 1
        mean_len = float(self.stats.get("mean_length", 0.0))
        mean_idx = float(self.stats.get("mean_critical_index", 0.0))
        self.stats["rollouts"] = rollouts
        self.stats["mean_length"] = mean_len + (length - mean_len) / rollouts
        self.stats["mean_critical_index"] = mean_idx + (index - mean_idx) / rollouts

    def reset_to(self, env: Any = None, critical: Any = None, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        env = env if env is not None else self.env
        critical = critical if critical is not None else self.last_critical_state
        if env is None:
            return np.zeros((0,), dtype=np.float32), {"restore_mode": "none"}
        if critical is None:
            result = env.reset()
            obs = result[0] if isinstance(result, (tuple, list)) and result else result
            return obs, {"restore_mode": "fallback"}
        payload = getattr(critical, "restore_payload", None)
        if payload is not None and hasattr(env, "reset_to_state"):
            try:
                out = env.reset_to_state(payload)
                if isinstance(out, tuple) and len(out) == 2:
                    return out[0], {**(out[1] if isinstance(out[1], dict) else {}), "restore_mode": "direct"}
                return out, {"restore_mode": "direct"}
            except Exception:
                pass
        if hasattr(env, "reset_to"):
            try:
                out = env.reset_to(critical, **kwargs)
                if isinstance(out, tuple) and len(out) == 2:
                    return out
                return out, {"restore_mode": "reset_to"}
            except Exception:
                pass
        try:
            result = env.reset()
        except TypeError:
            result = env.reset()
        obs = result[0] if isinstance(result, (tuple, list)) and result else result
        return obs, {"restore_mode": "fallback"}

    def reset_to_critical(self, env: Any = None, critical: Any = None, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        return self.reset_to(env=env, critical=critical, **kwargs)

    def summary(self) -> Dict[str, Any]:
        return {
            "name": "AIRS",
            "env_id": self.env_id,
            "K": self.K,
            "stats": dict(self.stats),
            "scorer": self.scorer.stats,
            "config": self.config.to_dict(),
        }

    @property
    def statistics(self) -> Dict[str, Any]:
        return dict(self.stats)


# ---------------------------------------------------------------------------
# Umbrella explainer (drop-in for RandomExplanation / IntegratedGradients)
# ---------------------------------------------------------------------------
class AIRS:
    """AIRS explanation baseline (attention-based) for RICE Experiment III.

    Wraps :class:`AIRSScorer` + :class:`AIRSStateSelector` behind the same
    interface as ``RandomExplanation`` / ``IntegratedGradients`` so that the
    explanation can be swapped while keeping the refining method fixed
    (Sec. 4.2 Experiment III / Appendix C.3 Table 6).
    """

    name = "AIRS"
    is_random = False

    def __init__(
        self,
        env: Any = None,
        policy: Any = None,
        env_id: str = "default",
        seed: Optional[int] = None,
        config: Optional[Any] = None,
        rng: Optional[Any] = None,
        K: Optional[int] = None,
        deterministic_policy: Optional[bool] = None,
        with_mask_shim: Optional[bool] = None,
        scorer: Optional[AIRSScorer] = None,
        selector: Optional[AIRSStateSelector] = None,
        observation_space: Any = None,
        action_space: Any = None,
        **kwargs: Any,
    ) -> None:
        self.config = config if isinstance(config, AIRSConfig) else AIRSConfig.from_dict(config or {})
        if env_id and env_id != "default":
            self.config.env_id = _normalize_env_key(env_id)
        self.env = env
        self.policy = policy
        self.env_id = self.config.env_id
        self.K = K if K is not None else self.config.K
        if deterministic_policy is not None:
            self.config.deterministic_policy = bool(deterministic_policy)
        if with_mask_shim is not None:
            self.config.with_mask_shim = bool(with_mask_shim)
        self.rng = rng if rng is not None else _get_rng(seed if seed is not None else self.config.seed)
        self.observation_space = observation_space
        self.action_space = action_space
        self._logger = _get_logger("rice.baselines.airs")

        self.network: Optional[Any] = None
        if _HAS_TORCH:
            obs_dim, action_dim = self._infer_dims(observation_space, action_space)
            if obs_dim and action_dim:
                try:
                    self.network = AIRSNetwork(
                        obs_dim=obs_dim,
                        action_dim=action_dim,
                        hidden_sizes=self.config.hidden_sizes,
                        activation=self.config.activation,
                        attention_mode=self.config.attention_mode,
                        temperature=self.config.temperature,
                        temporal=self.config.temporal,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    self._logger.debug("Could not build AIRS attention network: %s", exc)
                    self.network = None

        self.scorer = scorer or AIRSScorer(
            policy=policy,
            network=self.network,
            config=self.config,
            env_id=self.env_id,
            rng=self.rng,
            seed=self.config.seed,
            device=self.config.device,
        )
        self.selector = selector or AIRSStateSelector(
            env=env,
            policy=policy,
            mask_net=None,
            K=self.K,
            scorer=self.scorer,
            deterministic_policy=self.config.deterministic_policy,
            rng=self.rng,
            seed=self.config.seed,
            config=self.config,
        )
        self.air_network = self.network
        self.history: List[Any] = []
        self._logger.debug("Initialised AIRS explainer (%s)", describe_airs(self))

    @staticmethod
    def _infer_dims(observation_space: Any, action_space: Any) -> Tuple[Optional[int], Optional[int]]:
        obs_dim, action_dim = None, None
        shape = getattr(observation_space, "shape", None)
        if shape:
            obs_dim = int(np.prod(shape))
        if action_space is not None:
            shape = getattr(action_space, "shape", None)
            if shape:
                action_dim = int(np.prod(shape))
            elif hasattr(action_space, "n"):
                action_dim = int(action_space.n)
        return obs_dim, action_dim

    # -- training --------------------------------------------------------
    def fit(
        self,
        env: Any = None,
        policy: Any = None,
        observations: Any = None,
        n_samples: Optional[int] = None,
        epochs: Optional[int] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> "AIRS":
        """Train the interpretable attention module (optional).

        AIRS also works as a purely sensitivity-based explanation without
        training (uniform attention), but training the attention module sharpens
        the explanation and mirrors the original method.
        """
        if self.network is None or not _HAS_TORCH:
            self._logger.debug("AIRS attention training skipped (no torch/network)")
            return self
        env = env if env is not None else self.env
        policy = policy if policy is not None else self.policy
        if observations is None:
            if env is None or policy is None:
                return self
            observations = collect_policy_observations(
                env,
                policy,
                n_samples=n_samples or self.config.train_samples,
                seed=self.config.seed,
                deterministic=self.config.deterministic_policy,
            )
        self.network = train_airs_network(
            self.network,
            policy,
            observations,
            target=self.config.target,
            epochs=epochs or self.config.epochs,
            batch_size=self.config.batch_size,
            lr=self.config.lr,
            l1_coeff=self.config.l1_coeff,
            entropy_coeff=self.config.entropy_coeff,
            device=self.config.device,
            seed=self.config.seed,
            verbose=verbose,
        )
        self.air_network = self.network
        self.scorer.network = self.network
        return self

    # -- scoring ---------------------------------------------------------
    def score(self, observations: Any) -> np.ndarray:
        return self.scorer.score(observations)

    @property
    def importance_scores(self) -> Callable[[Any], np.ndarray]:
        return self.scorer.score

    def score_trajectory(self, trajectory: Any) -> Any:
        return self.scorer.score_trajectory(trajectory)

    def rank(self, trajectory: Any) -> np.ndarray:
        return self.scorer.rank(trajectory)

    def top_k_states(self, trajectory: Any, k: int = 10) -> List[Tuple[int, np.ndarray]]:
        return self.scorer.top_k_states(trajectory, k)

    def attach_to_wrapper(self, wrapper: Any, trajectory: Any = None) -> Any:
        return self.scorer.attach_to_wrapper(wrapper, trajectory)

    def __call__(self, trajectory: Any) -> np.ndarray:
        try:
            return self.scorer.importance_scores(trajectory)
        except Exception:
            return self.scorer.score(trajectory)

    # -- selection -------------------------------------------------------
    def rollout(self, policy: Any = None, K: Optional[int] = None, **kwargs: Any) -> Any:
        return self.selector.rollout(policy=policy, K=K, **kwargs)

    def select(self, policy: Any = None, K: Optional[int] = None, **kwargs: Any) -> Any:
        critical = self.selector.select(policy=policy, K=K, **kwargs)
        self.history.append(critical)
        return critical

    def select_top_k(self, k: int = 10, policy: Any = None, K: Optional[int] = None, **kwargs: Any) -> List[Any]:
        return self.selector.select_top_k(k=k, policy=policy, K=K, **kwargs)

    def identify(self, env: Any = None, policy: Any = None, K: Optional[int] = None, **kwargs: Any) -> Any:
        if env is not None:
            self.selector.env = env
        return self.select(policy=policy, K=K, **kwargs)

    identify_critical_state = identify

    def reset_to(self, env: Any = None, critical: Any = None, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        return self.selector.reset_to(env=env, critical=critical, **kwargs)

    def reset_to_critical(self, env: Any = None, critical: Any = None, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        return self.selector.reset_to_critical(env=env, critical=critical, **kwargs)

    # -- mask shim -------------------------------------------------------
    def as_mask_network(self, **kwargs: Any) -> Any:
        return _AIRSMaskShim(self.scorer, env_key=self.env_id)

    @property
    def mask_net(self) -> Any:
        return self.as_mask_network() if self.config.with_mask_shim else None

    @property
    def mask_network(self) -> Any:
        return self.mask_net

    def save(self, path: str) -> str:
        if self.network is None:
            raise RuntimeError("No AIRS attention network to save")
        return self.network.save(path, extra={"env_id": self.env_id, "config": self.config.to_dict()})

    def load(self, path: str) -> "AIRS":
        if not _HAS_TORCH:
            raise RuntimeError("AIRSNetwork is unavailable (torch missing)")
        self.network = AIRSNetwork.load(path, device=self.config.device)
        self.air_network = self.network
        self.scorer.network = self.network
        return self

    @property
    def statistics(self) -> Dict[str, Any]:
        return self.summary()

    @property
    def stats(self) -> Dict[str, Any]:
        return self.summary()

    def summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "env_id": self.env_id,
            "K": self.K,
            "score_mode": self.config.score_mode,
            "has_attention_network": self.network is not None,
            "scorer": self.scorer.stats,
            "selector": dict(self.selector.stats),
            "n_critical_states": len(self.history),
        }

    def describe(self) -> str:
        return describe_airs(self)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------
def identify_critical_state_with_airs(
    env: Any,
    policy: Any,
    K: Optional[int] = None,
    network: Optional[Any] = None,
    env_id: Optional[str] = None,
    deterministic: bool = True,
    reset: bool = True,
    seed: Optional[int] = None,
    return_rollout: bool = False,
    config: Optional[Any] = None,
    **kwargs: Any,
) -> Any:
    """Roll the frozen policy for K steps and return the AIRS-critical state."""
    resolved_env_id = env_id or (
        getattr(env, "rice_env_key", "default") if env is not None else "default"
    )
    airs = AIRS(
        env=env,
        policy=policy,
        env_id=resolved_env_id,
        seed=seed,
        config=config,
        K=K,
        deterministic_policy=deterministic,
        **kwargs,
    )
    if network is not None:
        airs.network = network
        airs.scorer.network = network
    rollout = airs.rollout(policy=policy, K=K, reset=reset, seed=seed)
    critical = airs.selector.select(policy=policy, K=K, reset=reset, seed=seed, rollout=rollout)
    if return_rollout:
        return critical, rollout
    return critical


def select_airs_states(
    env: Any,
    policy: Any,
    n: int = 1,
    K: Optional[int] = None,
    seed: Optional[int] = None,
    env_id: Optional[str] = None,
    deterministic: bool = True,
    **kwargs: Any,
) -> List[Any]:
    """``n`` independent rollouts, each contributing its AIRS-critical state."""
    airs = AIRS(env=env, policy=policy, env_id=env_id or "default", seed=seed, K=K, **kwargs)
    out: List[Any] = []
    for i in range(max(1, int(n))):
        out.append(airs.select(policy=policy, K=K, seed=None if seed is None else seed + i))
    return out


def make_airs(
    env: Any = None,
    policy: Any = None,
    env_id: Any = "default",
    seed: Optional[int] = None,
    config: Optional[Any] = None,
    K: Optional[int] = None,
    deterministic_policy: Optional[bool] = None,
    rng: Optional[Any] = None,
    **kwargs: Any,
) -> AIRS:
    """Factory creating a configured :class:`AIRS` explainer."""
    if isinstance(env_id, dict):  # tolerate (env, policy, config_dict) calling orders
        config = env_id
        env_id = "default"
    return AIRS(
        env=env,
        policy=policy,
        env_id=str(env_id),
        seed=seed,
        config=config,
        rng=rng,
        K=K,
        deterministic_policy=deterministic_policy,
        **kwargs,
    )


#: Aliases used by the baseline registry / drivers.
build_airs = make_airs
make_airs_explainer = make_airs


def airs_for(env_id: str = "default", **kwargs: Any) -> AIRS:
    """Factory pre-configured for a specific application id."""
    return make_airs(env_id=env_id, **kwargs)


def describe_airs(explanation: Any = None) -> str:
    """One-line human-readable description for logging / result tables."""
    if explanation is None:
        return "AIRS: attention-based explanation baseline (Yu et al., 2023)"
    config = getattr(explanation, "config", None)
    env_id = getattr(explanation, "env_id", getattr(config, "env_id", "default"))
    score_mode = getattr(config, "score_mode", "hybrid")
    engine = (
        "trained-attention"
        if getattr(explanation, "network", None) is not None
        else "feature-sensitivity"
    )
    return (
        f"AIRS explanation (env={env_id}, score_mode={score_mode}, engine={engine}); "
        "attention weights over state features -> step-level importance; "
        "used in Experiment III / Table 6 to show RICE still beats Random with other explanations"
    )


def samples_for(env_id: str, default: int = 300_000) -> int:
    """AIRS attention-module training-sample budget per application (Table 4)."""
    return int(MASK_SAMPLE_BUDGETS.get(_normalize_env_key(env_id), default))


__all__ = [
    # config
    "AIRSConfig",
    # network / training
    "AIRSNetwork",
    "train_airs_network",
    "collect_policy_observations",
    # scoring
    "airs_importance_scores",
    "airs_scores",
    # scorer / selector / umbrella
    "AIRSScorer",
    "AIRSStateSelector",
    "AIRS",
    # factories
    "make_airs",
    "build_airs",
    "make_airs_explainer",
    "airs_for",
    "identify_critical_state_with_airs",
    "select_airs_states",
    "describe_airs",
    "samples_for",
    # helpers
    "flatten_observation",
    "observation_matrix",
    "policy_action_output",
    # constants
    "DEFAULT_HIDDEN_SIZES",
    "DEFAULT_LR",
    "DEFAULT_EPOCHS",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_L1_COEFF",
    "DEFAULT_TOPK_FRACTION",
    "DEFAULT_K",
    "SCORE_MODES",
    "AIRS_TARGETS",
    "ATTENTION_MODES",
    "MASK_SAMPLE_BUDGETS",
]
