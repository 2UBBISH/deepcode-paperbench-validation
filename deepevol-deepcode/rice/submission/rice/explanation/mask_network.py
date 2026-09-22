"""Mask network (Stage-1 explanation) and the masked-action operator.

This module implements the *state mask* model of RICE (Cheng et al., ICML 2024,
PMLR 235, Section 3.3 "Step-level Explanation") together with the action-level
operator that turns mask decisions into an executed environment action::

    a_t (.) a_t^m = a_t        if a_t^m = 0
                    a_random   if a_t^m = 1

The mask network ``~pi_theta`` takes the target agent's current state ``s_t`` as
input and outputs a *binary* action ``a_t^m in {0, 1}`` (two logits / Bernoulli).
The mask is trained by Algorithm 1 with *vanilla PPO* on the augmented reward

    R'(s_t, a_t) = R(s_t, a_t) + alpha * a_t^m

(the additional bonus is required because naively maximizing ``eta(pi_bar)``
collapses to the trivial solution "never blind"; see Section 3.3).  The *state
importance* of ``s_t`` is the probability that the mask network outputs ``0``
("keep"):

    importance(s_t) = P(a_t^m = 0 | s_t) = softmax(logits(s_t))[keep_index].

Architectures follow the per-application table used by the target policy
(default ``MlpPolicy`` MLP for dense/sparse MuJoCo, ``[128, 128, 128, 128]`` for
Selfish Mining, ``[64, 64, 64]`` for CAGE-2, DI-engine VAC for MetaDrive).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..models import (
    MaskArch,
    build_mlp,
    mask_arch,
    normalize_env_key,
    sample_random_action,
)
from ..utils.io import ensure_dir

try:  # pragma: no cover - optional heavy dependency
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _HAS_TORCH = False


__all__ = [
    "MASK_KEEP",
    "MASK_BLIND",
    "KEEP_INDEX",
    "BLIND_INDEX",
    "NUM_MASK_ACTIONS",
    "MaskNetwork",
    "MaskCritic",
    "MaskedActionOperator",
    "build_mask_network",
    "build_mask_critic",
    "masked_action",
    "apply_mask",
    "masked_action_batch",
    "blind_mask",
    "augmented_reward",
    "blinding_bonus",
    "keep_probability_from_logits",
    "mask_from_logits",
    "state_importance",
    "mask_entropy",
    "save_mask_network",
    "load_mask_network",
    "describe_mask_network",
    "flatten_observation",
]

# ---------------------------------------------------------------------------
# Semantics of the binary mask action (paper Eq. (1))
# ---------------------------------------------------------------------------
MASK_KEEP: int = 0     # a_t^m = 0  ->  execute the target agent's action a_t
MASK_BLIND: int = 1    # a_t^m = 1  ->  execute a uniformly random action
KEEP_INDEX: int = MASK_KEEP
BLIND_INDEX: int = MASK_BLIND
NUM_MASK_ACTIONS: int = 2

_MASK_MODULE_BASE: Any = nn.Module if _HAS_TORCH else object


# ---------------------------------------------------------------------------
# Observation handling helpers
# ---------------------------------------------------------------------------
def flatten_observation(observation: Any) -> np.ndarray:
    """Flatten an observation (array-like, scalar, tuple or dict) to 1-D float32.

    Dict observations (used by the MetaDrive / CAGE-2 style applications) are
    concatenated over their sorted keys so that a single flat state vector can be
    fed to the mask network.
    """
    if observation is None:
        return np.zeros(0, dtype=np.float32)
    if isinstance(observation, dict):
        parts = [flatten_observation(observation[key]) for key in sorted(observation.keys())]
        return np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, dtype=np.float32)
    if _HAS_TORCH and isinstance(observation, torch.Tensor):  # pragma: no cover
        return observation.detach().cpu().numpy().reshape(-1).astype(np.float32)
    if isinstance(observation, (tuple, list)):
        try:
            return np.asarray(observation, dtype=np.float32).reshape(-1).astype(np.float32)
        except Exception:
            parts = [flatten_observation(o) for o in observation]
            return np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, dtype=np.float32)
    return np.asarray(observation, dtype=np.float32).reshape(-1).astype(np.float32)


def _infer_obs_dim(observation_space: Any = None, default: Optional[int] = None) -> int:
    """Best-effort inference of a flat observation dimensionality."""
    if observation_space is not None:
        shape = getattr(observation_space, "shape", None)
        if shape is not None:
            try:
                size = int(np.prod(shape))
                if size > 0:
                    return size
            except Exception:
                pass
        spaces = getattr(observation_space, "spaces", None)
        if isinstance(spaces, dict):
            total = sum(_infer_obs_dim(spaces[key], default=0) for key in sorted(spaces.keys()))
            if total > 0:
                return total
    if default is not None:
        return int(default)
    return 0


def _stack_observations(observations: Any) -> np.ndarray:
    """Normalise an observation container into a ``(N, D)`` float32 matrix."""
    if isinstance(observations, np.ndarray) and observations.dtype != object and observations.ndim >= 2:
        return observations.reshape(observations.shape[0], -1).astype(np.float32)
    if isinstance(observations, (list, tuple)) and len(observations) > 0:
        first = observations[0]
        if isinstance(first, (dict, list, tuple)) or np.ndim(first) > 0:
            return np.stack([flatten_observation(o) for o in observations], axis=0)
    return flatten_observation(observations).reshape(1, -1)


# ---------------------------------------------------------------------------
# Mask network
# ---------------------------------------------------------------------------
class MaskNetwork(_MASK_MODULE_BASE):  # type: ignore[misc]
    """Binary mask network ``~pi_theta(a_t^m | s_t)``.

    Consumes the state ``s_t`` and produces two logits over the mask actions
    ``{0 = keep, 1 = blind}``.  The *state importance* used throughout RICE is
    the softmax probability of the ``keep`` index.

    Parameters
    ----------
    obs_dim:
        Dimensionality of the (flattened) state input.
    hidden_sizes:
        Body hidden widths; defaults resolved from :func:`rice.models.mask_arch`.
    activation:
        Activation name (``"tanh"`` for the SB3-default policy network).
    env_key:
        Canonical environment key used for architecture resolution / metadata.
    actor:
        Optional externally provided module producing the 2 logits (e.g. an
        ``ActorCritic`` built with ``kind="mask"``).  When ``None`` a plain MLP
        head is built with :func:`rice.models.build_mlp`.
    device:
        Torch device string (``"cpu"`` by default).
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: str = "tanh",
        env_key: str = "default",
        num_logits: int = NUM_MASK_ACTIONS,
        keep_index: int = KEEP_INDEX,
        blind_index: int = BLIND_INDEX,
        device: str = "cpu",
        actor: Optional[Any] = None,
        init_orthogonal: bool = True,
        name: Optional[str] = None,
    ) -> None:
        if not _HAS_TORCH:  # pragma: no cover
            raise ImportError("MaskNetwork requires PyTorch (Stage-1 mask training).")
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self.activation = activation
        self.env_key = normalize_env_key(env_key)
        self.num_logits = int(num_logits)
        self.keep_index = int(keep_index)
        self.blind_index = int(blind_index)
        self.device = torch.device(device)
        self.name = name or f"mask_{self.env_key}"
        self.uses_external_actor = actor is not None

        if actor is not None:
            self.actor = actor
        else:
            self.actor = build_mlp(
                input_dim=self.obs_dim,
                hidden_sizes=self.hidden_sizes,
                output_dim=self.num_logits,
                activation=self.activation,
                output_activation=None,
                init_orthogonal=init_orthogonal,
            )
        self.to(self.device)

    # -- tensor conversion ---------------------------------------------------
    def _to_tensor(self, observation: Any) -> Any:
        if _HAS_TORCH and isinstance(observation, torch.Tensor):
            tensor = observation.to(self.device)
        else:
            tensor = torch.as_tensor(flatten_observation(observation), dtype=torch.float32, device=self.device)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    # -- forward -------------------------------------------------------------
    def forward(self, observation: Any) -> Any:
        """Raw 2-logit mask output, shape ``(B, 2)``."""
        obs = self._to_tensor(observation)
        out = self.actor(obs)
        if isinstance(out, tuple):
            out = out[0]
        out = out.reshape(obs.shape[0], -1)
        if out.shape[-1] != self.num_logits:
            # e.g. an ActorCritic-style module: recover logits from its Categorical.
            dist_fn = getattr(self.actor, "get_distribution", None)
            if callable(dist_fn):
                try:
                    logits = getattr(dist_fn(obs), "logits", None)
                    if logits is not None:
                        out = logits.reshape(obs.shape[0], -1)
                except Exception:
                    pass
        return out

    def logits(self, observation: Any) -> Any:
        return self.forward(observation)

    def distribution(self, observation: Any) -> Any:
        """Categorical distribution over the two mask actions."""
        return torch.distributions.Categorical(logits=self.forward(observation))

    def probabilities(self, observation: Any) -> Any:
        """Softmax probabilities ``P(a_t^m = . | s_t)`` of shape ``(B, 2)``."""
        with torch.no_grad():
            return F.softmax(self.forward(observation), dim=-1)

    # -- importance ----------------------------------------------------------
    def keep_probability(self, observation: Any) -> Any:
        """State importance ``P(a_t^m = 0 | s_t)`` as a tensor of shape ``(B,)``."""
        with torch.no_grad():
            probs = F.softmax(self.forward(observation), dim=-1)
            return probs[:, self.keep_index]

    def blind_probability(self, observation: Any) -> Any:
        with torch.no_grad():
            probs = F.softmax(self.forward(observation), dim=-1)
            return probs[:, self.blind_index]

    def score(self, observation: Any) -> np.ndarray:
        """Numpy state-importance scores ``P(keep)`` for one or many states."""
        with torch.no_grad():
            scores = self.keep_probability(observation)
        return scores.detach().cpu().numpy().reshape(-1).astype(np.float32)

    # -- sampling ------------------------------------------------------------
    def sample(self, observation: Any, deterministic: bool = False) -> Tuple[Any, Any, Any]:
        """Sample mask actions, as ``theta_old`` does in Algorithm 1.

        Returns ``(masks, log_prob, entropy)`` with ``masks`` an int64 tensor of
        shape ``(B,)`` taking values in ``{0, 1}``.
        """
        logits = self.forward(observation)
        dist = torch.distributions.Categorical(logits=logits)
        masks = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        return masks, dist.log_prob(masks), dist.entropy()

    def greedy(self, observation: Any) -> np.ndarray:
        """Most likely mask action(s) (argmax) as int64 numpy array."""
        with torch.no_grad():
            masks = torch.argmax(self.forward(observation), dim=-1)
        return masks.detach().cpu().numpy().reshape(-1).astype(np.int64)

    def act(self, observation: Any, deterministic: bool = False) -> Any:
        """Single mask action (int) or a batch (numpy array)."""
        if deterministic:
            masks = self.greedy(observation)
        else:
            with torch.no_grad():
                masks, _, _ = self.sample(observation, deterministic=False)
            masks = masks.detach().cpu().numpy().reshape(-1).astype(np.int64)
        return int(masks[0]) if masks.size == 1 else masks

    def evaluate_actions(self, observation: Any, masks: Any) -> Tuple[Any, Any]:
        """Return ``(log_prob, entropy)`` of given mask actions (PPO ratio term)."""
        dist = self.distribution(observation)
        if not (_HAS_TORCH and isinstance(masks, torch.Tensor)):
            masks = torch.as_tensor(np.asarray(masks), dtype=torch.long, device=self.device)
        masks = masks.to(self.device).reshape(-1).long()
        return dist.log_prob(masks), dist.entropy()

    def get_parameters(self) -> Iterable[Any]:
        return self.actor.parameters()

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"env_key={self.env_key}, obs_dim={self.obs_dim}, "
            f"hidden={list(self.hidden_sizes)}, activation={self.activation!r}, "
            f"keep_index={self.keep_index}"
        )


class MaskCritic(_MASK_MODULE_BASE):  # type: ignore[misc]
    """State-value baseline for the vanilla PPO update of Algorithm 1.

    The mask network only parameterises ``~pi_theta(a_t^m | s_t)``; PPO still needs
    a value baseline to form advantages.  The critic is not part of the explanation
    itself (it is unused at evaluation time) and therefore lives in its own module.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: str = "tanh",
        device: str = "cpu",
    ) -> None:
        if not _HAS_TORCH:  # pragma: no cover
            raise ImportError("MaskCritic requires PyTorch.")
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self.device = torch.device(device)
        self.net = build_mlp(
            input_dim=self.obs_dim,
            hidden_sizes=self.hidden_sizes,
            output_dim=1,
            activation=activation,
            output_activation=None,
            init_orthogonal=True,
        )
        self.to(self.device)

    def forward(self, observation: Any) -> Any:
        if _HAS_TORCH and isinstance(observation, torch.Tensor):
            obs = observation.to(self.device)
        else:
            obs = torch.as_tensor(flatten_observation(observation), dtype=torch.float32, device=self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.net(obs).reshape(-1)

    def predict_values(self, observation: Any) -> Any:
        with torch.no_grad():
            return self.forward(observation)


# ---------------------------------------------------------------------------
# Masked-action operator (paper Eq. 1)
# ---------------------------------------------------------------------------
def _as_int_mask(mask: Any) -> np.ndarray:
    """Normalise any mask representation to an int64 numpy array over ``{0, 1}``."""
    if _HAS_TORCH and isinstance(mask, torch.Tensor):  # pragma: no cover
        arr = mask.detach().cpu().numpy()
    else:
        arr = np.asarray(mask)
    arr = arr.reshape(-1)
    if arr.dtype == bool:
        return arr.astype(np.int64)
    if arr.dtype.kind == "f":
        # Floating masks are read as "blind whenever >= 0.5".
        return (arr >= 0.5).astype(np.int64)
    return arr.astype(np.int64)


def masked_action(
    action: Any,
    mask: Any,
    action_space: Any = None,
    rng: Optional[Any] = None,
    dim: Optional[int] = None,
    discrete: bool = False,
    low: Optional[Any] = None,
    high: Optional[Any] = None,
    dtype: Any = np.float32,
) -> np.ndarray:
    """Execute the mask operator of Eq. (1).

    ``a_t (.) a_t^m = a_t`` when ``a_t^m == 0`` and ``a_random`` (uniform over the
    action space) when ``a_t^m == 1``.  Supports both single actions and batches
    (``action`` of shape ``(B, A)`` with ``mask`` of shape ``(B,)``).
    """
    target = np.asarray(action, dtype=np.float32)
    masks = _as_int_mask(mask)

    if dim is None:
        if target.ndim >= 1 and target.shape[-1] > 0:
            dim = int(target.shape[-1])
        elif action_space is not None and getattr(action_space, "shape", None) is not None:
            dim = int(np.prod(action_space.shape))
        else:
            dim = 1

    if not discrete and action_space is not None and hasattr(action_space, "n"):
        discrete = True

    def _rand(n: int) -> np.ndarray:
        samples = [
            np.asarray(
                sample_random_action(
                    action_space=action_space,
                    rng=rng,
                    dim=dim,
                    discrete=bool(discrete),
                    low=low,
                    high=high,
                    dtype=dtype,
                ),
                dtype=np.float32,
            ).reshape(-1)
            for _ in range(max(int(n), 1))
        ]
        return np.stack(samples, axis=0)

    # Scalar / single-step case.
    if target.ndim == 1 and masks.size == 1:
        if int(masks[0]) == MASK_BLIND:
            return _rand(1)[0].reshape(-1).astype(dtype)
        return target.reshape(-1).astype(dtype)

    # Batched case.
    n = int(masks.size)
    if target.ndim == 1:
        target = np.repeat(target.reshape(1, -1), n, axis=0)
    target = target.reshape(n, -1)
    random_actions = _rand(n)
    blind = (masks == MASK_BLIND).reshape(-1, 1)
    return np.where(blind, random_actions, target).astype(dtype)


# SB3-compatible aliases ----------------------------------------------------
apply_mask = masked_action
masked_action_batch = masked_action


def blind_mask(mask: Any) -> np.ndarray:
    """Boolean array marking steps where the target agent is blinded."""
    return _as_int_mask(mask) == MASK_BLIND


def augmented_reward(rewards: Any, masks: Any, alpha: float = 1e-4, return_tensor: bool = False) -> Any:
    """``R'(s_t, a_t) = R(s_t, a_t) + alpha * a_t^m`` (Section 3.3)."""
    r = np.asarray(rewards, dtype=np.float32).reshape(-1)
    m = _as_int_mask(masks).astype(np.float32).reshape(-1)
    if m.size != r.size:
        m = m[: r.size] if m.size > r.size else np.pad(m, (0, r.size - m.size))
    out = r + float(alpha) * m
    if return_tensor and _HAS_TORCH:  # pragma: no cover
        return torch.as_tensor(out, dtype=torch.float32)
    return out


def blinding_bonus(masks: Any, alpha: float = 1e-4) -> np.ndarray:
    """The bonus term ``alpha * a_t^m`` alone (logging / diagnostics)."""
    return (float(alpha) * _as_int_mask(masks).astype(np.float32)).reshape(-1)


# ---------------------------------------------------------------------------
# logits-level helpers (numpy / torch agnostic)
# ---------------------------------------------------------------------------
def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim == 1:
        logits = logits.reshape(1, -1)
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def keep_probability_from_logits(logits: Any, keep_index: int = KEEP_INDEX) -> np.ndarray:
    """``P(a_t^m = 0 | s_t)`` computed directly from logits."""
    return _softmax(logits)[:, int(keep_index)].reshape(-1)


def mask_from_logits(logits: Any, keep_index: int = KEEP_INDEX, deterministic: bool = True) -> np.ndarray:
    """Mask action(s) from logits: greedy argmax, or Bernoulli sampling of P(blind)."""
    probs = _softmax(logits)
    if deterministic:
        return np.argmax(probs, axis=-1).astype(np.int64)
    blind_index = 1 - int(keep_index) if probs.shape[-1] == NUM_MASK_ACTIONS else probs.shape[-1] - 1
    return (np.random.rand(probs.shape[0]) < probs[:, blind_index]).astype(np.int64)


def state_importance(mask_net: "MaskNetwork", observations: Any, batch_size: int = 4096) -> np.ndarray:
    """Score states with the trained mask net: ``importance = P(keep)``."""
    obs_matrix = _stack_observations(observations)
    scores: List[np.ndarray] = []
    for start in range(0, obs_matrix.shape[0], max(int(batch_size), 1)):
        chunk = obs_matrix[start : start + batch_size]
        scores.append(mask_net.score(chunk))
    if not scores:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(scores, axis=0).astype(np.float32)


def mask_entropy(mask_net: "MaskNetwork", observations: Any, batch_size: int = 4096) -> float:
    """Mean Bernoulli entropy (nats) of the mask distribution.

    Low entropy means the mask is decisive (keeps or blinds with confidence) — a
    useful diagnostic for Stage-1 training.
    """
    if not _HAS_TORCH:  # pragma: no cover
        return float("nan")
    obs_matrix = _stack_observations(observations)
    entropies: List[float] = []
    with torch.no_grad():
        for start in range(0, obs_matrix.shape[0], max(int(batch_size), 1)):
            chunk = obs_matrix[start : start + batch_size]
            entropies.append(float(mask_net.distribution(chunk).entropy().mean().item()))
    return float(np.mean(entropies)) if entropies else float("nan")


# ---------------------------------------------------------------------------
# Operator object bundling policy + mask + action space
# ---------------------------------------------------------------------------
class MaskedActionOperator:
    """Couples a frozen target policy ``pi`` with a mask net ``~pi_theta``.

    ``act()`` samples the target action ``a_t ~ pi(.|s_t)`` and the mask action
    ``a_t^m ~ ~pi_theta(.|s_t)`` and returns the executed action ``a_t (.) a_t^m``
    together with bookkeeping (mask, state importance) used by the fidelity
    evaluator and the refinement loop.
    """

    def __init__(
        self,
        mask_net: "MaskNetwork",
        action_space: Any = None,
        env_id: str = "default",
        discrete: Optional[bool] = None,
        deterministic_mask: bool = False,
        seed: Optional[int] = None,
        rng: Optional[Any] = None,
    ) -> None:
        self.mask_net = mask_net
        self.env_id = normalize_env_key(env_id)
        self.action_space = action_space
        self.deterministic_mask = bool(deterministic_mask)
        self.rng = rng if rng is not None else np.random.RandomState(seed)
        if discrete is None:
            discrete = bool(action_space is not None and hasattr(action_space, "n"))
        self.discrete = bool(discrete)
        self.num_steps = 0
        self.num_blinded = 0
        shape = getattr(action_space, "shape", None)
        self.action_dim = int(np.prod(shape)) if shape is not None else 1
        self.low = getattr(action_space, "low", None)
        self.high = getattr(action_space, "high", None)

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _split_policy_output(output: Any) -> np.ndarray:
        """Accept ``action`` or ``(action, state)`` from a policy call."""
        if isinstance(output, tuple):
            output = output[0]
        return np.asarray(output, dtype=np.float32)

    def sample_target_action(self, policy: Any, observation: Any, deterministic: bool = False) -> np.ndarray:
        if policy is None:
            raise ValueError("MaskedActionOperator.act requires a policy or an explicit target_action")
        output = (
            policy.predict(observation, deterministic=deterministic)
            if hasattr(policy, "predict")
            else policy.act(observation, deterministic=deterministic)
        )
        return self._split_policy_output(output)

    # -- main entry point ----------------------------------------------------
    def act(
        self,
        observation: Any,
        target_action: Any = None,
        policy: Any = None,
        deterministic_mask: Optional[bool] = None,
        deterministic_policy: bool = False,
    ) -> Dict[str, Any]:
        """Return the executed action plus mask bookkeeping for one step."""
        det_mask = self.deterministic_mask if deterministic_mask is None else bool(deterministic_mask)
        if target_action is None:
            target_action = self.sample_target_action(policy, observation, deterministic=deterministic_policy)
        target_action = np.asarray(target_action, dtype=np.float32).reshape(-1)

        if _HAS_TORCH:
            with torch.no_grad():
                mask_t, _, _ = self.mask_net.sample(observation, deterministic=det_mask)
                importance = self.mask_net.keep_probability(observation)
            mask = int(mask_t.detach().cpu().numpy().reshape(-1)[0])
            importance = float(importance.detach().cpu().numpy().reshape(-1)[0])
        else:  # pragma: no cover - a mask net cannot exist without torch
            mask, importance = MASK_KEEP, 1.0

        executed = masked_action(
            target_action,
            mask,
            action_space=self.action_space,
            rng=self.rng,
            dim=self.action_dim,
            discrete=self.discrete,
            low=self.low,
            high=self.high,
        )
        self.num_steps += 1
        self.num_blinded += int(mask == MASK_BLIND)
        return {
            "action": executed.reshape(-1).astype(np.float32),
            "target_action": target_action.astype(np.float32),
            "mask": int(mask),
            "importance": float(importance),
            "blinded": bool(mask == MASK_BLIND),
        }

    # -- diagnostics ---------------------------------------------------------
    @property
    def blind_fraction(self) -> float:
        return float(self.num_blinded) / float(self.num_steps) if self.num_steps else 0.0

    def stats(self) -> Dict[str, float]:
        return {
            "steps": float(self.num_steps),
            "blinded": float(self.num_blinded),
            "blind_fraction": self.blind_fraction,
        }

    def reset(self) -> None:
        self.num_steps = 0
        self.num_blinded = 0


# ---------------------------------------------------------------------------
# Factories / persistence / descriptions
# ---------------------------------------------------------------------------
def build_mask_critic(
    env_id: str = "default",
    obs_dim: Optional[int] = None,
    observation_space: Any = None,
    hidden_sizes: Optional[Sequence[int]] = None,
    activation: Optional[str] = None,
    device: str = "cpu",
) -> "MaskCritic":
    """Build the PPO value baseline used during Stage-1 mask training."""
    arch: MaskArch = mask_arch(env_id, obs_dim=obs_dim)
    dim = int(obs_dim or arch.obs_dim or _infer_obs_dim(observation_space))
    return MaskCritic(
        obs_dim=dim,
        hidden_sizes=hidden_sizes or arch.hidden_sizes,
        activation=activation or arch.activation,
        device=device,
    )


def build_mask_network(
    env_id: str = "default",
    obs_dim: Optional[int] = None,
    action_dim: Optional[int] = None,
    observation_space: Any = None,
    action_space: Any = None,
    hidden_sizes: Optional[Sequence[int]] = None,
    activation: Optional[str] = None,
    device: str = "cpu",
    actor: Optional[Any] = None,
    keep_index: int = KEEP_INDEX,
    blind_index: int = BLIND_INDEX,
    **kwargs: Any,
) -> "MaskNetwork":
    """Instantiate a mask network following the per-application architecture."""
    arch: MaskArch = mask_arch(env_id, obs_dim=obs_dim)
    dim = int(obs_dim or arch.obs_dim or _infer_obs_dim(observation_space))
    net = MaskNetwork(
        obs_dim=dim,
        hidden_sizes=hidden_sizes or arch.hidden_sizes,
        activation=activation or arch.activation,
        env_key=env_id,
        device=device,
        actor=actor,
        keep_index=keep_index,
        blind_index=blind_index,
    )
    # Metadata used for logging / checkpoint round-trips.
    net.action_dim = int(action_dim) if action_dim is not None else None
    return net


def save_mask_network(
    mask_net: "MaskNetwork",
    path: str,
    env_id: Optional[str] = None,
    obs_dim: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
    **meta: Any,
) -> str:
    """Persist mask-network weights + metadata to ``path`` (creating dirs)."""
    if not _HAS_TORCH:  # pragma: no cover
        raise ImportError("save_mask_network requires PyTorch.")
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        ensure_dir(directory)
    payload: Dict[str, Any] = {
        "state_dict": mask_net.state_dict(),
        "env_key": env_id or mask_net.env_key,
        "obs_dim": int(obs_dim or mask_net.obs_dim),
        "hidden_sizes": list(mask_net.hidden_sizes),
        "activation": mask_net.activation,
        "keep_index": int(mask_net.keep_index),
        "blind_index": int(mask_net.blind_index),
        "kind": "mask",
    }
    if extra:
        payload.update(extra)
    payload.update(meta)
    torch.save(payload, path)
    return path


def load_mask_network(
    path: str,
    env_id: Optional[str] = None,
    obs_dim: Optional[int] = None,
    observation_space: Any = None,
    device: str = "cpu",
    strict: bool = False,
    **kwargs: Any,
) -> "MaskNetwork":
    """Rebuild a mask network from a checkpoint written by :func:`save_mask_network`."""
    if not _HAS_TORCH:  # pragma: no cover
        raise ImportError("load_mask_network requires PyTorch.")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(f"Not a mask-network checkpoint: {path}")
    key = env_id or ckpt.get("env_key", "default")
    dim = int(obs_dim or ckpt.get("obs_dim") or _infer_obs_dim(observation_space))
    net = MaskNetwork(
        obs_dim=dim,
        hidden_sizes=tuple(ckpt.get("hidden_sizes", (64, 64))),
        activation=ckpt.get("activation", "tanh"),
        env_key=key,
        device=device,
        keep_index=int(ckpt.get("keep_index", KEEP_INDEX)),
        blind_index=int(ckpt.get("blind_index", BLIND_INDEX)),
    )
    try:
        net.load_state_dict(ckpt["state_dict"], strict=strict)
    except Exception:
        net.load_state_dict(ckpt["state_dict"], strict=False)
    net.eval()
    return net


def describe_mask_network(mask_net: "MaskNetwork") -> str:
    """Human-readable architecture summary (used in logs and READMEs)."""
    return (
        f"MaskNetwork(env_key={mask_net.env_key}, obs_dim={mask_net.obs_dim}, "
        f"hidden_sizes={list(mask_net.hidden_sizes)}, activation={mask_net.activation!r}, "
        f"num_logits={mask_net.num_logits}, keep_index={mask_net.keep_index}, "
        f"blind_index={mask_net.blind_index}, device={mask_net.device})"
    )
