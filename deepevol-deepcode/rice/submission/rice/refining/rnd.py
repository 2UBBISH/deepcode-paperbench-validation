"""Random Network Distillation (RND) module for RICE Stage 2.

Paper reference (RICE, ICML 2024, Section 3.3 "Exploration with Random Network
Distillation" and Algorithm 2)::

    Calculate RND bonus  R_t^{RND} = || f(s_{t+1}) - \hat{f}(s_{t+1}) ||^2
    with normalization
    Add (s_t, s_{t+1}, a_t, R_t + \lambda R_t^{RND}) to D
    ...
    Optimize \hat{f}_\theta w.r.t. MSE loss on D using Adam

    "we optimize  R'(s_t, a_t) = R(s_t, a_t) + \lambda | f(s_{t+1}) - \hat{f}(s_{t+1}) |^2,
     where \lambda controls the trade-off between the task reward and exploration
     bonus.  Along with the policy parameters, the RND predictor network \hat{f} is
     updated to regress to the target network f.  Note that, as the state coverage
     increases, RND bonuses decay to zero and a performed policy is recovered."

The paper does **not** specify the network sizes nor the exact normalization form
(the plan's documented defaults are used instead, see the module docstring of
``rice/refining/__init__.py``):

* target ``f`` is fixed after random (orthogonal) initialization,
* predictor ``f_hat`` has the *same shape* as ``f`` and is trained with Adam + MSE,
* the intrinsic reward is normalized by the **running standard deviation of the
  prediction error** (Burda et al., 2018 style), implemented with
  :class:`rice.utils.metrics.RunningMeanStd`,
* the default network is a small MLP (2 x 64 tanh), matching the SB3 MlpPolicy body.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - torch is a hard runtime dependency of the RICE pipeline
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = object  # type: ignore
    F = None  # type: ignore
    _HAS_TORCH = False

try:  # optional: reuse shared statistics machinery
    from ..utils.metrics import RunningMeanStd  # type: ignore
except Exception:  # pragma: no cover
    RunningMeanStd = None  # type: ignore


__all__ = [
    "RND_DEFAULT_HIDDEN",
    "DEFAULT_LAMBDA",
    "RNDConfig",
    "RNDNetwork",
    "RNDModule",
    "RNDBonus",
    "build_rnd_network",
    "build_rnd",
    "normalize_bonus",
    "rnd_bonus",
    "make_rnd",
]


# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

#: Default RND network width. The paper leaves the sizes unspecified; the plan's
#: documented default is "a small MLP".  2 x 64 tanh matches the SB3 ``MlpPolicy``.
RND_DEFAULT_HIDDEN: Tuple[int, ...] = (64, 64)

#: Default trade-off coefficient lambda between task reward and RND bonus.
DEFAULT_LAMBDA: float = 0.01


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


@dataclass
class RNDConfig:
    """Configuration bundle for the RND intrinsic-reward module.

    Fields
    ------
    obs_dim:
        Dimension of the (flattened) observation fed to ``f`` / ``f_hat``.
    hidden_sizes:
        Hidden layer widths for both the target and the predictor networks.
    activation:
        Activation name (``"tanh"``, ``"relu"``, ``"gelu"``, ``"elu"``).
    learning_rate / lr:
        Adam learning rate for the predictor.
    lam / lambda_:
        Trade-off coefficient ``lambda`` added to the task reward.
    normalize:
        Whether to divide the intrinsic reward by the running std of the error.
    clip_bonus:
        Optional upper clip for the (normalized) bonus; ``None`` disables it.
    update_epochs / batch_size:
        Number of passes / minibatch size when regressing ``f_hat`` onto ``f``.
    reward_scale:
        Extra multiplicative factor applied after normalization.
    device:
        Torch device string.
    """

    obs_dim: int = 0
    hidden_sizes: Sequence[int] = field(default_factory=lambda: tuple(RND_DEFAULT_HIDDEN))
    activation: str = "tanh"
    learning_rate: float = 1e-4
    lam: float = DEFAULT_LAMBDA
    normalize: bool = True
    clip_bonus: Optional[float] = None
    update_epochs: int = 1
    batch_size: int = 256
    reward_scale: float = 1.0
    device: str = "cpu"

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]] = None, **overrides: Any) -> "RNDConfig":
        cfg = dict(cfg or {})
        # accept nested {"rnd": {...}} / {"refining": {...}} config sections
        for key in ("rnd", "rnd_config", "refining", "exploration"):
            nested = cfg.get(key)
            if isinstance(nested, dict):
                merged = dict(nested)
                merged.update({k: v for k, v in cfg.items() if k != key})
                cfg = merged
                break
        # alias handling (the paper writes lambda; CLI/YAML may use "lambda")
        aliases = {
            "lambda": "lam",
            "lambda_": "lam",
            "lam": "lam",
            "lr": "learning_rate",
            "learning_rate": "learning_rate",
            "hidden": "hidden_sizes",
            "hidden_sizes": "hidden_sizes",
            "net_arch": "hidden_sizes",
        }
        kwargs: Dict[str, Any] = {}
        for key, value in cfg.items():
            target = aliases.get(key, key)
            if target in cls.__dataclass_fields__:
                kwargs[target] = value
        for key, value in overrides.items():
            target = aliases.get(key, key)
            if target in cls.__dataclass_fields__ and value is not None:
                kwargs[target] = value
        obj = cls(**kwargs)
        obj.hidden_sizes = tuple(int(h) for h in obj.hidden_sizes)
        return obj

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["hidden_sizes"] = list(self.hidden_sizes)
        d["lambda"] = self.lam
        return d

    # ---------------------------------------------------------------------- properties
    @property
    def lr(self) -> float:  # pragma: no cover - convenience alias
        return self.learning_rate

    @property
    def lambda_(self) -> float:  # pragma: no cover - convenience alias
        return self.lam


# --------------------------------------------------------------------------------------
# Networks
# --------------------------------------------------------------------------------------


def _get_activation(name: str):
    name = (name or "tanh").lower()
    if not _HAS_TORCH:
        return name
    mapping = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
        "silu": nn.SiLU,
    }
    return mapping.get(name, nn.Tanh)


class RNDNetwork(nn.Module):  # type: ignore[misc]
    """Small MLP mapping an observation to a feature vector.

    Used for both the fixed, randomly-initialized target ``f`` and the trainable
    predictor ``f_hat`` (they share the shape, not the weights).  Orthogonal
    initialization (SB3 style) keeps the target's output distribution stable.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Sequence[int] = RND_DEFAULT_HIDDEN,
        activation: str = "tanh",
        output_dim: Optional[int] = None,
    ) -> None:
        if not _HAS_TORCH:
            raise ImportError("RNDNetwork requires PyTorch, which is not available.")
        super().__init__()
        hidden_sizes = tuple(int(h) for h in hidden_sizes) or ()
        self.obs_dim = int(obs_dim)
        self.hidden_sizes = hidden_sizes
        self.activation_name = activation
        # feature size equals the last hidden width (or obs_dim when no hidden layers)
        self.output_dim = int(output_dim if output_dim is not None else (hidden_sizes[-1] if hidden_sizes else self.obs_dim))

        act_cls = _get_activation(activation)
        layers: List[nn.Module] = []
        last = self.obs_dim
        for width in hidden_sizes:
            lin = nn.Linear(last, int(width))
            nn.init.orthogonal_(lin.weight, gain=np.sqrt(2))
            nn.init.constant_(lin.bias, 0.0)
            layers += [lin, act_cls()]
            last = int(width)
        if not hidden_sizes:
            lin = nn.Linear(self.obs_dim, self.output_dim)
            nn.init.orthogonal_(lin.weight, gain=np.sqrt(2))
            nn.init.constant_(lin.bias, 0.0)
            layers.append(lin)
        self.net = nn.Sequential(*layers)

    def forward(self, obs) -> Any:  # type: ignore[override]
        if not _HAS_TORCH:
            raise ImportError("RNDNetwork requires PyTorch, which is not available.")
        x = obs
        if not torch.is_tensor(x):
            x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        x = x.float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)

    def features(self, obs) -> np.ndarray:
        """Numpy front-end returning ``f(obs)`` as a 2D ``np.ndarray``."""
        if not _HAS_TORCH:
            raise ImportError("RNDNetwork requires PyTorch, which is not available.")
        with torch.no_grad():
            out = self.forward(obs)
        return out.detach().cpu().numpy()

    def extra_repr(self) -> str:  # pragma: no cover - debug helper
        return f"obs_dim={self.obs_dim}, hidden_sizes={tuple(self.hidden_sizes)}, out={self.output_dim}"


def build_rnd_network(
    obs_dim: int,
    hidden_sizes: Sequence[int] = RND_DEFAULT_HIDDEN,
    activation: str = "tanh",
    output_dim: Optional[int] = None,
) -> "RNDNetwork":
    """Construct a single RND MLP (target or predictor)."""
    return RNDNetwork(obs_dim, hidden_sizes=hidden_sizes, activation=activation, output_dim=output_dim)


# --------------------------------------------------------------------------------------
# Bonus normalization helpers
# --------------------------------------------------------------------------------------


def normalize_bonus(
    errors: np.ndarray,
    rms: Optional[Any] = None,
    clip: Optional[float] = None,
    update: bool = True,
    eps: float = 1e-8,
) -> np.ndarray:
    """Normalize raw prediction errors by their running standard deviation.

    The paper only says "with normalization"; the plan's documented default is to
    divide the intrinsic reward by the running std of the prediction error
    (Burda et al., 2018).  ``rms`` must expose ``update`` and ``std`` (i.e. be a
    :class:`rice.utils.metrics.RunningMeanStd`); when it is unavailable, the
    batch-wise std is used as a fallback so the bonus stays scale-invariant.
    """
    err = np.asarray(errors, dtype=np.float64).reshape(-1)
    if not update:
        # normalization without touching the running statistics
        if rms is not None:
            std = float(getattr(rms, "std", 1.0) or 1.0)
        else:
            std = float(np.std(err)) if err.size > 1 else 1.0
        out = err / (std + eps)
        return out if clip is None else np.clip(out, 0.0, clip)

    if rms is not None and hasattr(rms, "update") and hasattr(rms, "std"):
        rms.update(err.reshape(-1, 1))
        std = float(rms.std) if np.isfinite(rms.std) else 1.0
    else:
        std = float(np.std(err)) if err.size > 1 else 1.0
    out = err / (max(std, eps))
    if clip is not None:
        out = np.clip(out, 0.0, clip)
    return out


def rnd_bonus(
    target_features: np.ndarray,
    predictor_features: np.ndarray,
    normalization: str = "running_std",
    rms: Optional[Any] = None,
    update_stats: bool = True,
    clip: Optional[float] = None,
    scale: float = 1.0,
    raw_errors: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute ``|| f(s) - f_hat(s) ||^2`` (optionally normalized).

    Parameters
    ----------
    target_features, predictor_features:
        ``f(s)`` and ``f_hat(s)`` of identical shape ``(batch, feature_dim)`` or
        ``(feature_dim,)``.
    normalization:
        ``"running_std"`` (default, plan's documented default), ``"batch_std"``,
        ``"raw"``/``"none"``.
    rms:
        Running-statistics object used for ``running_std`` normalization.
    """
    f = np.asarray(target_features, dtype=np.float64)
    f_hat = np.asarray(predictor_features, dtype=np.float64)
    if f.shape != f_hat.shape:
        f_hat = np.broadcast_to(f_hat, f.shape)
    diff = f - f_hat
    if diff.ndim == 1:
        squared = float(np.sum(diff ** 2))
        errors = np.asarray([squared], dtype=np.float64)
        squeeze = True
    else:
        errors = np.sum(diff ** 2, axis=-1).reshape(-1).astype(np.float64)
        squeeze = False

    mode = (normalization or "none").lower()
    if mode in ("none", "raw", "off", "false"):
        out = errors
    elif mode == "batch_std":
        std = float(np.std(errors)) if errors.size > 1 else 1.0
        out = errors / (std + 1e-8)
    else:  # running_std (default)
        out = normalize_bonus(errors, rms=rms, clip=None, update=update_stats)

    out = out * float(scale)
    if clip is not None:
        out = np.clip(out, 0.0, clip)
    return float(out[0]) if squeeze else out  # type: ignore[index]


# --------------------------------------------------------------------------------------
# Main RND module
# --------------------------------------------------------------------------------------


class RNDModule:
    """Random Network Distillation intrinsic-reward module (Algorithm 2).

    Owns:

    * ``target`` : frozen, randomly initialized network ``f``,
    * ``predictor`` : trainable network ``f_hat`` regressed onto ``f`` with Adam/MSE,
    * ``rms`` : running statistics of the prediction error used for normalization.

    Typical use inside the refinement loop::

        rnd = RNDModule(obs_dim)
        bonus = rnd.bonus(s_next)               # after env.step
        rnd.update([(s_next, None)])            # optimizer step(s)

    ``RNDModule`` never touches the policy: it only supplies the additive term
    ``lambda * R_t^RND`` of the paper's augmented reward.
    """

    def __init__(
        self,
        obs_dim: Optional[int] = None,
        hidden_sizes: Sequence[int] = RND_DEFAULT_HIDDEN,
        activation: str = "tanh",
        learning_rate: float = 1e-4,
        lam: float = DEFAULT_LAMBDA,
        normalize: bool = True,
        clip_bonus: Optional[float] = None,
        update_epochs: int = 1,
        batch_size: int = 256,
        reward_scale: float = 1.0,
        device: str = "cpu",
        config: Optional[Any] = None,
        seed: Optional[int] = None,
        observation_space: Any = None,
        obs_dim_: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if not _HAS_TORCH:
            raise ImportError("RNDModule requires PyTorch, which is not available.")

        if config is not None:
            cfg = RNDConfig.from_dict(config if isinstance(config, dict) else getattr(config, "__dict__", {}))
            obs_dim = obs_dim if obs_dim is not None else cfg.obs_dim
            hidden_sizes = cfg.hidden_sizes
            activation = cfg.activation
            learning_rate = cfg.learning_rate
            lam = cfg.lam
            normalize = cfg.normalize
            clip_bonus = cfg.clip_bonus
            update_epochs = cfg.update_epochs
            batch_size = cfg.batch_size
            reward_scale = cfg.reward_scale
            device = cfg.device

        if obs_dim is None:
            obs_dim = obs_dim_
        if obs_dim is None and observation_space is not None:
            shape = getattr(observation_space, "shape", None)
            if shape is not None:
                obs_dim = int(np.prod(shape))
        # tolerate extra YAML keys (e.g. "refining/lam" collisions)
        obs_dim = int(obs_dim or kwargs.get("observation_dim") or 0)
        if obs_dim <= 0:
            raise ValueError("RNDModule requires a positive obs_dim (or an observation_space).")

        self.device = torch.device(device or "cpu")
        self.obs_dim = obs_dim
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self.lam = float(lam)
        self.normalize = bool(normalize)
        self.clip_bonus = clip_bonus
        self.reward_scale = float(reward_scale)
        self.update_epochs = max(1, int(update_epochs))
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.seed = seed

        if seed is not None:
            torch.manual_seed(int(seed))

        self.target = RNDNetwork(obs_dim, self.hidden_sizes, activation).to(self.device)
        self.predictor = RNDNetwork(obs_dim, self.hidden_sizes, activation).to(self.device)

        # target ``f`` is FIXED after random initialization (Burda et al., 2018)
        for param in self.target.parameters():
            param.requires_grad_(False)
        self.target.eval()

        # predictor starts as a copy of the target?  No: RND needs a non-zero initial
        # error signal, so the predictor keeps its own independent random init.
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=self.learning_rate)

        if RunningMeanStd is not None:
            self.rms = RunningMeanStd(shape=(1,))
        else:  # minimal internal fallback (Welford mean/var over the error)
            self.rms = _FallbackRMS()

        # bookkeeping
        self.update_count = 0
        self.bonus_calls = 0
        self._error_mean = 0.0
        self._error_var = 1.0
        self._error_count = 0
        self._last_bonus: float = 0.0

    # ------------------------------------------------------------------- features
    def _to_tensor(self, obs: Any):
        x = obs if torch.is_tensor(obs) else torch.as_tensor(np.asarray(obs), dtype=torch.float32)
        x = x.float().to(self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x

    def target_features(self, obs: Any) -> np.ndarray:
        """Frozen target features ``f(obs)`` as numpy (2D)."""
        with torch.no_grad():
            return self.target(self._to_tensor(obs)).detach().cpu().numpy()

    def predictor_features(self, obs: Any) -> np.ndarray:
        """Predictor features ``f_hat(obs)`` as numpy (2D)."""
        with torch.no_grad():
            return self.predictor(self._to_tensor(obs)).detach().cpu().numpy()

    features = target_features  # pragma: no cover - alias

    # --------------------------------------------------------------------- bonus
    def raw_error(self, obs: Any) -> np.ndarray:
        """Unnormalized squared prediction error ``||f(s) - f_hat(s)||^2``."""
        with torch.no_grad():
            x = self._to_tensor(obs)
            f = self.target(x)
            f_hat = self.predictor(x)
            err = torch.sum((f - f_hat) ** 2, dim=-1)
            return err.detach().cpu().numpy().reshape(-1)

    def bonus(self, obs: Any, update_stats: bool = True) -> np.ndarray:
        """Intrinsic reward ``R_t^{RND}`` for one or more next states.

        Matches Algorithm 2: ``R_t^RND = ||f(s_{t+1}) - f_hat(s_{t+1})||^2`` with
        normalization.  Returns a float for a single state and an array for a batch.
        """
        err = self.raw_error(obs)
        self.bonus_calls += int(err.size)
        if not self.normalize:
            out = err * self.reward_scale
        else:
            out = normalize_bonus(err, rms=self.rms, clip=self.clip_bonus, update=update_stats)
            out = out * self.reward_scale
        out = np.asarray(out, dtype=np.float64).reshape(-1)
        self._last_bonus = float(out[0]) if out.size else 0.0
        # track raw error statistics for diagnostics
        self._error_mean = float(np.mean(err)) if err.size else self._error_mean
        self._error_var = float(np.var(err)) if err.size else self._error_var
        self._error_count += int(err.size)
        return float(out[0]) if out.size == 1 else out

    def __call__(self, obs: Any, update_stats: bool = True) -> np.ndarray:
        return self.bonus(obs, update_stats=update_stats)

    def augmented_reward(
        self,
        task_reward: float,
        next_obs: Any,
        lam: Optional[float] = None,
        update_stats: bool = True,
    ) -> Tuple[float, float]:
        """Return ``(task_reward + lambda * bonus, bonus)`` for Algorithm 2 line 12."""
        bonus = self.bonus(next_obs, update_stats=update_stats)
        bonus = float(np.asarray(bonus).reshape(-1)[0])
        lam = self.lam if lam is None else float(lam)
        return float(task_reward) + lam * bonus, bonus

    # -------------------------------------------------------------------- updates
    def update(self, observations: Any, epochs: Optional[int] = None) -> Dict[str, float]:
        """Regress ``f_hat`` onto ``f`` with MSE loss via Adam (Algorithm 2 line 15).

        ``observations`` may be the next states collected in ``D`` (an array, a list
        of arrays, a list of ``(s_t, s_{t+1})`` tuples, or a list of transitions).
        """
        obs = self._as_observation_batch(observations)
        if obs is None or len(obs) == 0:
            return {"rnd/loss": 0.0, "rnd/n_updates": 0.0, "rnd/mean_error": 0.0}

        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        n = obs_t.shape[0]
        epochs = self.update_epochs if epochs is None else max(1, int(epochs))
        batch_size = min(self.batch_size, max(1, n)) if self.batch_size else n
        losses: List[float] = []

        self.predictor.train()
        for _ in range(epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                batch = obs_t[idx]
                with torch.no_grad():
                    target_out = self.target(batch)
                pred_out = self.predictor(batch)
                loss = F.mse_loss(pred_out, target_out)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(float(loss.detach().cpu().item()))
        self.predictor.eval()
        self.update_count += 1
        return {
            "rnd/loss": float(np.mean(losses)) if losses else 0.0,
            "rnd/n_updates": float(len(losses)),
            "rnd/mean_error": float(np.mean(losses)) if losses else 0.0,
        }

    def update_from_dataset(self, dataset: Iterable[Any]) -> Dict[str, float]:
        """Update from the PPO dataset ``D`` (accepts transitions with next states)."""
        next_states = []
        for item in dataset:
            # (s_t, s_{t+1}, a_t, R_t)  /  (s_t, s_{t+1})  /  dict
            if isinstance(item, dict):
                nxt = item.get("next_observation", item.get("next_obs", item.get("s_next")))
                if nxt is not None:
                    next_states.append(np.asarray(nxt, dtype=np.float32).reshape(-1))
                continue
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                nxt = item[1]
                nxt = np.concatenate([np.asarray(x, dtype=np.float32).reshape(-1) for x in (nxt if isinstance(nxt, (list, tuple)) else [nxt])])
                next_states.append(nxt)
                continue
            next_states.append(np.asarray(item, dtype=np.float32).reshape(-1))
        if not next_states:
            return {"rnd/loss": 0.0, "rnd/n_updates": 0.0, "rnd/mean_error": 0.0}
        return self.update(np.asarray(next_states, dtype=np.float32))

    def _as_observation_batch(self, observations: Any) -> Optional[np.ndarray]:
        if observations is None:
            return None
        arr = observations
        if isinstance(arr, np.ndarray):
            arr = np.asarray(arr, dtype=np.float32)
            return arr.reshape(-1, self.obs_dim) if arr.ndim != 2 else arr
        if torch.is_tensor(arr):
            arr = arr.detach().cpu().numpy()
            return np.asarray(arr, dtype=np.float32).reshape(-1, self.obs_dim)
        if isinstance(arr, (list, tuple)):
            if len(arr) == 0:
                return np.zeros((0, self.obs_dim), dtype=np.float32)
            if isinstance(arr[0], (tuple, list)) and len(arr[0]) >= 2:
                # transitions: take the next state
                rows = [np.asarray(item[1], dtype=np.float32).reshape(-1) for item in arr]
            else:
                rows = [np.asarray(item, dtype=np.float32).reshape(-1) for item in arr]
            return np.asarray(rows, dtype=np.float32)
        return np.asarray(arr, dtype=np.float32).reshape(-1, self.obs_dim)

    # ------------------------------------------------------------------ utilities
    def statistics(self) -> Dict[str, float]:
        """Diagnostics; ``rnd/bonus`` decays to zero as coverage grows."""
        return {
            "rnd/lambda": float(self.lam),
            "rnd/mean_error": float(self._error_mean),
            "rnd/std_error": float(np.sqrt(max(self._error_var, 0.0))),
            "rnd/last_bonus": float(self._last_bonus),
            "rnd/n_bonus_calls": float(self.bonus_calls),
            "rnd/n_updates": float(self.update_count),
            "rnd/running_std": float(getattr(self.rms, "std", 1.0)) if self.rms is not None else 1.0,
        }

    stats = statistics  # pragma: no cover - alias

    def state_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target.state_dict(),
            "predictor": self.predictor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rms": _rms_state(self.rms),
            "config": {
                "obs_dim": self.obs_dim,
                "hidden_sizes": list(self.hidden_sizes),
                "lam": self.lam,
                "normalize": self.normalize,
                "clip_bonus": self.clip_bonus,
                "learning_rate": self.learning_rate,
                "reward_scale": self.reward_scale,
            },
            "counters": {"update_count": self.update_count, "bonus_calls": self.bonus_calls},
        }

    def load_state_dict(self, state: Dict[str, Any], strict: bool = False) -> "RNDModule":
        if "target" in state:
            self.target.load_state_dict(state["target"], strict=strict)
        if "predictor" in state:
            self.predictor.load_state_dict(state["predictor"], strict=strict)
        if "optimizer" in state:
            try:
                self.optimizer.load_state_dict(state["optimizer"])
            except Exception:  # pragma: no cover - optimizer mismatch is non fatal
                pass
        _set_rms_state(self.rms, state.get("rms"))
        counters = state.get("counters", {})
        self.update_count = int(counters.get("update_count", self.update_count))
        self.bonus_calls = int(counters.get("bonus_calls", self.bonus_calls))
        return self

    def save(self, path: str, extra: Optional[Dict[str, Any]] = None) -> str:
        if not _HAS_TORCH:
            raise ImportError("RNDModule.save requires PyTorch.")
        try:
            from ..utils.io import ensure_dir  # type: ignore
        except Exception:  # pragma: no cover
            def ensure_dir(p: str) -> str:  # type: ignore
                os.makedirs(os.path.dirname(os.path.abspath(p)) or ".", exist_ok=True)
                return p

        directory = os.path.dirname(os.path.abspath(path))
        ensure_dir(directory)
        payload = self.state_dict()
        if extra:
            payload["extra"] = extra
        torch.save(payload, path)
        return path

    @classmethod
    def load(cls, path: str, device: str = "cpu", obs_dim: Optional[int] = None, **build_kwargs: Any) -> "RNDModule":
        if not _HAS_TORCH:
            raise ImportError("RNDModule.load requires PyTorch.")
        state = torch.load(path, map_location=device)
        cfg = state.get("config", {})
        module = cls(
            obs_dim=obs_dim or cfg.get("obs_dim"),
            hidden_sizes=cfg.get("hidden_sizes", RND_DEFAULT_HIDDEN),
            learning_rate=cfg.get("learning_rate", 1e-4),
            lam=cfg.get("lam", DEFAULT_LAMBDA),
            normalize=cfg.get("normalize", True),
            clip_bonus=cfg.get("clip_bonus"),
            reward_scale=cfg.get("reward_scale", 1.0),
            device=device,
            **build_kwargs,
        )
        module.load_state_dict(state)
        return module

    def set_lambda(self, lam: float) -> None:
        self.lam = float(lam)

    def reset_statistics(self) -> None:
        if self.rms is not None and hasattr(self.rms, "reset"):
            try:
                self.rms.reset()
            except Exception:  # pragma: no cover
                self.rms = RunningMeanStd(shape=(1,))
        self._error_mean = 0.0
        self._error_var = 1.0
        self._error_count = 0
        self._last_bonus = 0.0

    def extra_repr(self) -> str:  # pragma: no cover - debug helper
        return (
            f"obs_dim={self.obs_dim}, hidden={tuple(self.hidden_sizes)}, "
            f"lambda={self.lam}, normalize={self.normalize}"
        )


class _FallbackRMS:
    """Minimal streaming mean/std tracker used when ``utils.metrics`` is missing."""

    def __init__(self) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return
        batch_mean = float(np.mean(arr))
        batch_var = float(np.var(arr))
        batch_count = arr.size
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta ** 2 * self.count * batch_count / total) / total
        self.count = total

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var))

    def reset(self) -> None:
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4


def _rms_state(rms: Any) -> Optional[Dict[str, Any]]:
    if rms is None:
        return None
    state = {}
    for key in ("mean", "var", "count"):
        if hasattr(rms, key):
            state[key] = np.asarray(getattr(rms, key)).tolist()
    return state or None


def _set_rms_state(rms: Any, state: Optional[Dict[str, Any]]) -> None:
    if rms is None or not state:
        return
    for key, value in state.items():
        if hasattr(rms, key):
            try:
                setattr(rms, key, np.asarray(value))
            except Exception:  # pragma: no cover
                pass


# --------------------------------------------------------------------------------------
# Convenience wrappers
# --------------------------------------------------------------------------------------


def build_rnd(
    obs_dim: Optional[int] = None,
    observation_space: Any = None,
    hidden_sizes: Sequence[int] = RND_DEFAULT_HIDDEN,
    activation: str = "tanh",
    learning_rate: float = 1e-4,
    lam: float = DEFAULT_LAMBDA,
    normalize: bool = True,
    clip_bonus: Optional[float] = None,
    update_epochs: int = 1,
    batch_size: int = 256,
    reward_scale: float = 1.0,
    device: str = "cpu",
    config: Optional[Any] = None,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> RNDModule:
    """Factory for :class:`RNDModule` (accepts an ``observation_space`` directly)."""
    return RNDModule(
        obs_dim=obs_dim,
        hidden_sizes=hidden_sizes,
        activation=activation,
        learning_rate=learning_rate,
        lam=lam,
        normalize=normalize,
        clip_bonus=clip_bonus,
        update_epochs=update_epochs,
        batch_size=batch_size,
        reward_scale=reward_scale,
        device=device,
        config=config,
        seed=seed,
        observation_space=observation_space,
        **kwargs,
    )


make_rnd = build_rnd  # pragma: no cover - alias


@dataclass
class RNDBonus:
    """Stateless helper computing normalized RND bonuses from raw feature arrays.

    Useful when the target/predictor forward passes happen elsewhere (e.g. inside a
    vectorized rollout), keeping the ``||f(s)-f_hat(s)||^2`` contract in one place.
    """

    lam: float = DEFAULT_LAMBDA
    normalization: str = "running_std"
    clip: Optional[float] = None
    scale: float = 1.0

    def __call__(
        self,
        target_features: np.ndarray,
        predictor_features: np.ndarray,
        rms: Optional[Any] = None,
        update_stats: bool = True,
    ) -> np.ndarray:
        bonus = rnd_bonus(
            target_features,
            predictor_features,
            normalization=self.normalization,
            rms=rms,
            update_stats=update_stats,
            clip=self.clip,
            scale=self.scale,
        )
        arr = np.asarray(bonus, dtype=np.float64).reshape(-1)
        return arr * self.lam
