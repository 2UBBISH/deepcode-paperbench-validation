"""Random Network Distillation (RND) exploration bonus for RICE.

Paper reference
---------------
RICE (ICML 2024, PMLR 235), §3.3 "Technique Detail (Exploration with Random
Network Distillation)", verbatim:

    "... we directly utilize the PPO algorithm to update the policy ``pi``,
    except that we add the intrinsic reward to the task reward, that is, we
    optimize

        R'(s_t, a_t) = R(s_t, a_t) + lambda * |f(s_{t+1}) - f_hat(s_{t+1})|^2,

    where ``lambda`` controls the trade-off between the task reward and
    exploration bonus.  Along with the policy parameters, the RND predictor
    network ``f_hat`` is updated to regress to the target network ``f``.  Note
    that, as the state coverage increases, RND bonuses decay to zero and a
    performed policy is recovered."

and Algorithm 2, verbatim:

    "Calculate RND bonus  R_t^{RND} = || f(s_{t+1}) - f_hat(s_{t+1}) ||^2
     with normalization"
    ...
    "Optimize f_hat_theta w.r.t. MSE loss on D using Adam"

Design notes (Burda et al. 2018 recipe, paper unspecified details)
------------------------------------------------------------------
* ``f`` (the *target*) is a randomly initialized network whose weights are
  **frozen**; ``f_hat`` (the *predictor*) has the same architecture and is
  trained with Adam + MSE to regress the target's outputs.
* The intrinsic reward is computed on the **next** state ``s_{t+1}`` (exactly as
  written in Eq. of §3.3 / Algorithm 2), and the predictor is likewise trained
  on the visited ``s_{t+1}`` states stored in the rollout buffer.
* "with normalization": following Burda et al. (2018) we (a) normalize
  observations with running mean/std (clipped), and (b) divide the raw
  prediction error by a running standard deviation of observed errors, clipping
  the result.  Both running statistics are *stateful* and are stored in the
  checkpoint, so a resumed run continues where it left off.
* The bonus magnitude naturally decays to zero as the predictor catches up,
  i.e. as the state coverage grows; ``RND.mean_error`` / ``RND.history`` expose
  this for tests and figures.

Nothing here depends on the target agent's internals, which preserves the paper's
black-box assumption: we only ever consume (batches of) observations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

try:  # torch is a hard dependency of the algorithms layer (see ppo.py)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - informative failure mode only
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _TORCH_AVAILABLE = False

from .ppo import flatten_obs, observation_size, resolve_device


__all__ = [
    "RunningMeanStd",
    "RNDNetwork",
    "RNDConfig",
    "RND",
    "compute_rnd_bonus",
    "make_rnd",
]


# ---------------------------------------------------------------------------
# Running statistics
# ---------------------------------------------------------------------------
class RunningMeanStd:
    """Numerically stable running mean/variance (Welford / parallel update).

    Used for two things in RND (Burda et al. 2018):
      * normalizing observations (``clip`` typically 5.0),
      * normalizing the intrinsic reward, i.e. dividing the raw prediction error
        by a running std of the errors observed so far (``clip`` typically 10.0).

    Kept in pure NumPy so that it can be saved/loaded independently of torch and
    so it works for scalar (reward) and vector (observation) statistics.
    """

    def __init__(
        self,
        shape: Union[int, Sequence[int]] = (),
        epsilon: float = 1e-4,
        clip: Optional[float] = None,
    ) -> None:
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = tuple(shape)
        self.epsilon = float(epsilon)
        self.clip = clip
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.var = np.ones(self.shape, dtype=np.float64)
        self.count = float(self.epsilon)

    # -- core update --------------------------------------------------------
    def update(self, x: Union[np.ndarray, float]) -> "RunningMeanStd":
        """Update statistics with a batch of samples.

        ``x`` may have shape ``self.shape`` (a single sample) or
        ``(batch, *self.shape)``.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == len(self.shape) and self.shape and x.shape != self.shape:
            # e.g. shape=() statistics receiving a scalar-shaped array
            x = x.reshape(self.shape)
        if x.size == 0:
            return self
        if not self.shape:
            x = x.reshape(-1)
            batch_mean = float(np.mean(x)) if x.size else 0.0
            batch_var = float(np.var(x)) if x.size else 0.0
            batch_count = float(x.size)
        else:
            x = x.reshape(-1, *self.shape)
            batch_mean = x.mean(axis=0)
            batch_var = x.var(axis=0)
            batch_count = float(x.shape[0])

        self._update_from_moments(batch_mean, batch_var, batch_count)
        return self

    def _update_from_moments(
        self,
        batch_mean: Union[np.ndarray, float],
        batch_var: Union[np.ndarray, float],
        batch_count: float,
    ) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        if tot_count <= 0:
            return
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = np.asarray(batch_var) * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.mean = new_mean
        self.var = m2 / tot_count
        self.count = tot_count

    # -- consumer API -------------------------------------------------------
    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var)

    def normalize(
        self,
        x: Union[np.ndarray, float],
        clip: Optional[float] = None,
        epsilon: Optional[float] = None,
        update: bool = False,
    ) -> np.ndarray:
        """Return ``(x - mean) / sqrt(var + eps)`` optionally clipped."""
        x = np.asarray(x, dtype=np.float64)
        if update:
            self.update(x)
        eps = self.epsilon if epsilon is None else float(epsilon)
        out = (x - self.mean) / np.sqrt(self.var + eps)
        clip_value = self.clip if clip is None else clip
        if clip_value is not None:
            out = np.clip(out, -float(clip_value), float(clip_value))
        return out

    def scale(self, x: Union[np.ndarray, float], epsilon: float = 1e-8) -> np.ndarray:
        """Divide by the running std only (used to normalize the RND bonus)."""
        x = np.asarray(x, dtype=np.float64)
        return x / np.sqrt(self.var + epsilon)

    # -- serialization ------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {
            "shape": self.shape,
            "epsilon": self.epsilon,
            "clip": self.clip,
            "mean": self.mean,
            "var": self.var,
            "count": self.count,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> "RunningMeanStd":
        self.shape = tuple(state.get("shape", self.shape))
        self.epsilon = float(state.get("epsilon", self.epsilon))
        self.clip = state.get("clip", self.clip)
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state.get("count", self.count))
        return self

    def copy(self) -> "RunningMeanStd":
        new = RunningMeanStd(self.shape, self.epsilon, self.clip)
        new.load_state_dict(self.state_dict())
        return new


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------
_DEFAULT_ACTIVATION = "relu"


def _activation_module(name: str):
    name = (name or _DEFAULT_ACTIVATION).lower()
    table = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "leaky_relu": nn.LeakyReLU,
        "silu": nn.SiLU,
        "sigmoid": nn.Sigmoid,
    }
    if name not in table:
        raise ValueError(f"Unsupported activation '{name}'. Options: {sorted(table)}")
    return table[name]


def _init_ortho(module, gain: float = np.sqrt(2)) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class RNDNetwork(nn.Module):
    """Plain MLP used for both the frozen target ``f`` and the predictor ``f_hat``.

    Deliberately architecture-agnostic w.r.t. the target agent (black-box RICE):
    it only consumes flattened observations, as in Burda et al. (2018).
    """

    def __init__(
        self,
        input_dim: int,
        net_arch: Sequence[int] = (64, 64),
        output_dim: int = 64,
        activation: str = _DEFAULT_ACTIVATION,
        ortho_init: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        act_cls = _activation_module(activation)

        layers: list = []
        last = self.input_dim
        for hidden in net_arch:
            layers.append(nn.Linear(last, int(hidden)))
            layers.append(act_cls())
            last = int(hidden)
        self.feature = nn.Sequential(*layers)
        self.head = nn.Linear(last, self.output_dim)

        if ortho_init:
            for m in self.feature:
                _init_ortho(m)
            _init_ortho(self.head, gain=1.0)

    def forward(self, obs) -> "torch.Tensor":
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(np.asarray(obs, dtype=np.float32))
        obs = obs.float()
        return self.head(self.feature(obs))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class RNDConfig:
    """Hyper-parameters of the RND module.

    The paper does not specify these (Source: not specified in the paper), so
    the documented defaults follow Burda et al. (2018) / common practice and are
    deliberately small/fast.  ``lambda`` (the intrinsic-reward coefficient) is a
    *refining-loop* hyper-parameter (Table 3); it is only stored here so that
    :meth:`RND.augmented_reward` can be called without an explicit coefficient.
    """

    net_arch: Tuple[int, ...] = (64, 64)
    predictor_arch: Optional[Tuple[int, ...]] = None  # defaults to net_arch
    target_arch: Optional[Tuple[int, ...]] = None  # defaults to net_arch
    output_dim: int = 64
    activation: str = _DEFAULT_ACTIVATION
    ortho_init: bool = True

    predictor_learning_rate: float = 1e-3
    predictor_betas: Tuple[float, float] = (0.9, 0.999)
    predictor_epsilon: float = 1e-5
    predictor_weight_decay: float = 0.0

    normalize_obs: bool = True
    clip_obs: float = 5.0
    obs_epsilon: float = 1e-8
    obs_stats_from_pretrain: bool = False

    normalize_reward: bool = True
    clip_reward: Optional[float] = 10.0
    reward_epsilon: float = 1e-8

    #: reduction over the feature dimension of ``(f - f_hat)^2``
    reduction: str = "mean"  # {"mean", "sum"}

    #: intrinsic-reward coefficient lambda (see Table 3); None -> must be passed
    coef: Optional[float] = 0.01

    update_epochs: int = 1
    batch_size: int = 256
    max_grad_norm: Optional[float] = 0.5

    device: str = "auto"
    seed: Optional[int] = None
    verbose: int = 0

    def clone(self, **overrides) -> "RNDConfig":
        return replace(self, **overrides)


# ---------------------------------------------------------------------------
# RND module
# ---------------------------------------------------------------------------
class RND:
    """Random Network Distillation exploration bonus.

    Typical use inside Algorithm 2 (see ``refine.py``)::

        rnd = RND(observation_space=env.observation_space, config=cfg)
        ...
        # per environment step, paper Eq.: bonus computed on s_{t+1}
        bonus = rnd.intrinsic_reward(next_obs)                     # normalized
        buffer.add(obs, action, reward + lam * float(bonus), next_obs, done, ...)
        ...
        # once per outer iteration (Algorithm 2)
        rnd.update_from_buffer(buffer)                              # Adam + MSE

    Parameters
    ----------
    observation_space:
        Gym/Gymnasium space (used to infer the input dimension). Optional if
        ``obs_dim`` is given.
    obs_dim:
        Explicit input dimension (overrides ``observation_space``).
    config:
        :class:`RNDConfig`; ``kwargs`` override individual fields.
    """

    def __init__(
        self,
        observation_space: Any = None,
        obs_dim: Optional[int] = None,
        config: Optional[RNDConfig] = None,
        device: str = "auto",
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        if not _TORCH_AVAILABLE:  # pragma: no cover
            raise ImportError("RND requires PyTorch. Install torch to use RND.")

        cfg = config.clone(**kwargs) if (config is not None and kwargs) else (config or RNDConfig(**kwargs))
        self.config = cfg

        if obs_dim is None:
            if observation_space is None:
                raise ValueError("Provide either `observation_space` or `obs_dim`.")
            obs_dim = observation_size(observation_space)
        self.obs_dim = int(obs_dim)

        self.device = resolve_device(device if device != "auto" else cfg.device)

        if seed is None:
            seed = cfg.seed
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed) % (2 ** 32))

        target_arch = cfg.target_arch or cfg.net_arch
        predictor_arch = cfg.predictor_arch or cfg.net_arch
        self.target = RNDNetwork(
            self.obs_dim,
            net_arch=target_arch,
            output_dim=cfg.output_dim,
            activation=cfg.activation,
            ortho_init=cfg.ortho_init,
        ).to(self.device)
        self.predictor = RNDNetwork(
            self.obs_dim,
            net_arch=predictor_arch,
            output_dim=cfg.output_dim,
            activation=cfg.activation,
            ortho_init=cfg.ortho_init,
        ).to(self.device)

        # frozen target: no grads, eval mode forever
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.target.eval()

        self.optimizer = torch.optim.Adam(
            self.predictor.parameters(),
            lr=cfg.predictor_learning_rate,
            betas=cfg.predictor_betas,
            eps=cfg.predictor_epsilon,
            weight_decay=cfg.predictor_weight_decay,
        )

        # normalizers
        self.obs_rms = RunningMeanStd(shape=(self.obs_dim,), clip=cfg.clip_obs)
        self.reward_rms = RunningMeanStd(shape=(), clip=None)

        # bookkeeping for tests / figures
        self.history: list = []  # per-update statistics
        self.update_count = 0
        self._last_mean_error = float("nan")
        self._last_mean_bonus = float("nan")

    # ------------------------------------------------------------------ obs
    def preprocess_observation(
        self,
        obs: Any,
        update_stats: bool = False,
        normalize: Optional[bool] = None,
    ) -> np.ndarray:
        """Flatten (+ optionally normalize) observations into float32 arrays."""
        flat = np.asarray(flatten_obs(obs), dtype=np.float32)
        if flat.ndim == 1:
            flat = flat.reshape(1, -1)
        normalize = self.config.normalize_obs if normalize is None else normalize
        if not normalize:
            return flat
        if update_stats:
            self.obs_rms.update(flat)
        return self.obs_rms.normalize(flat, clip=self.config.clip_obs).astype(np.float32)

    # --------------------------------------------------------------- forward
    def _target_output(self, obs_tensor: "torch.Tensor") -> "torch.Tensor":
        with torch.no_grad():
            return self.target(obs_tensor)

    def _predictor_output(self, obs_tensor: "torch.Tensor") -> "torch.Tensor":
        return self.predictor(obs_tensor)

    def _to_tensor(self, obs: Any) -> "torch.Tensor":
        arr = np.asarray(obs, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return torch.as_tensor(arr, device=self.device)

    def squared_error(
        self,
        obs: Any,
        normalize_obs: Optional[bool] = None,
        update_stats: bool = False,
    ) -> np.ndarray:
        """Raw per-sample ``(f - f_hat)^2`` reduced over the output dimension.

        Returns a 1-D array of length ``batch``.  This is the *un-normalized*
        quantity inside the paper's ``R_t^{RND}``.
        """
        obs_np = self.preprocess_observation(obs, update_stats=update_stats, normalize=normalize_obs)
        obs_t = self._to_tensor(obs_np)
        with torch.no_grad():
            target_out = self._target_output(obs_t)
            predictor_out = self._predictor_output(obs_t)
            diff = (target_out - predictor_out) ** 2
            if self.config.reduction == "sum":
                err = diff.sum(dim=-1)
            else:
                err = diff.mean(dim=-1)
        return err.detach().cpu().numpy().astype(np.float64).reshape(-1)

    def error(self, obs: Any, **kwargs) -> np.ndarray:
        """Alias of :meth:`squared_error` (friendly name for tests)."""
        return self.squared_error(obs, **kwargs)

    # -------------------------------------------------------- intrinsic reward
    def intrinsic_reward(
        self,
        obs: Any,
        normalize: Optional[bool] = None,
        update_stats: bool = True,
        normalize_obs: Optional[bool] = None,
    ) -> np.ndarray:
        """``R^{RND}`` for a batch of states -- paper Eq. with normalization.

        ``normalize=None`` -> use ``config.normalize_reward``.  The normalizer's
        running statistics are updated with the *raw* errors of this batch when
        ``update_stats`` is True (standard RND behaviour).
        """
        cfg = self.config
        normalize = cfg.normalize_reward if normalize is None else bool(normalize)
        raw = self.squared_error(obs, normalize_obs=normalize_obs, update_stats=True)
        self._last_mean_error = float(np.mean(raw)) if raw.size else float("nan")
        if normalize:
            self.reward_rms.update(raw)
            bonus = self.reward_rms.scale(raw, epsilon=cfg.reward_epsilon)
            if cfg.clip_reward is not None:
                bonus = np.clip(bonus, 0.0, float(cfg.clip_reward))
            # guard against pathological non-finite values
            bonus = np.nan_to_num(bonus, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            bonus = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        self._last_mean_bonus = float(np.mean(bonus)) if np.size(bonus) else float("nan")
        return bonus.astype(np.float32)

    def bonus(self, obs: Any, **kwargs) -> np.ndarray:
        """Alias of :meth:`intrinsic_reward`."""
        return self.intrinsic_reward(obs, **kwargs)

    def scaled_bonus(
        self,
        obs: Any,
        coef: Optional[float] = None,
        update_stats: bool = True,
        normalize: Optional[bool] = None,
    ) -> np.ndarray:
        """``lambda * R^{RND}`` (``coef`` defaults to ``config.coef``)."""
        c = self.config.coef if coef is None else float(coef)
        return c * self.intrinsic_reward(obs, normalize=normalize, update_stats=update_stats)

    def augmented_reward(
        self,
        rewards: Union[float, np.ndarray],
        next_obs: Any,
        coef: Optional[float] = None,
        normalize: Optional[bool] = None,
        update_stats: bool = True,
    ) -> np.ndarray:
        """Paper Eq.: ``R'(s_t,a_t) = R(s_t,a_t) + lambda * R^{RND}(s_{t+1})``.

        Returns the augmented reward (a float for scalar input).
        """
        c = self.config.coef if coef is None else float(coef)
        bonus = self.intrinsic_reward(next_obs, normalize=normalize, update_stats=update_stats)
        r = np.asarray(rewards, dtype=np.float64)
        out = r.reshape(-1) + c * np.asarray(bonus, dtype=np.float64).reshape(-1)
        return out

    # ----------------------------------------------------------- predictor fit
    def update(
        self,
        obs: Any,
        normalize_obs: Optional[bool] = None,
        update_stats: bool = False,
        n_updates: Optional[int] = None,
        batch_size: Optional[int] = None,
        return_losses: bool = False,
    ) -> Union[float, np.ndarray]:
        """Regress ``f_hat`` to ``f`` with MSE via Adam (Algorithm 2 last line).

        ``obs`` is the batch of states stored in the rollout buffer (the paper
        trains on ``s_{t+1}``).  Returns the final mean MSE loss (or the list of
        per-step losses if ``return_losses``).
        """
        cfg = self.config
        obs_np = self.preprocess_observation(obs, update_stats=update_stats, normalize=normalize_obs)
        n = obs_np.shape[0]
        if n == 0:
            return 0.0 if not return_losses else np.zeros(0, dtype=np.float32)

        bs = int(batch_size or cfg.batch_size)
        bs = max(1, min(bs, n))
        epochs = int(n_updates if n_updates is not None else cfg.update_epochs)
        epochs = max(1, epochs)

        obs_t = torch.as_tensor(obs_np, device=self.device)
        self.predictor.train()

        losses: list = []
        for _ in range(epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, bs):
                idx = perm[start : start + bs]
                batch = obs_t[idx]
                with torch.no_grad():
                    target_out = self.target(batch)
                pred_out = self.predictor(batch)
                if self.config.reduction == "sum":
                    loss = F.mse_loss(pred_out, target_out, reduction="sum") / batch.shape[0]
                else:
                    loss = F.mse_loss(pred_out, target_out, reduction="mean")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.predictor.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                losses.append(float(loss.detach().cpu().item()))

        self.update_count += 1
        mean_loss = float(np.mean(losses)) if losses else 0.0
        self.history.append(
            {
                "update": self.update_count,
                "mse": mean_loss,
                "mean_error": float(self._last_mean_error),
                "mean_bonus": float(self._last_mean_bonus),
                "reward_std": float(np.sqrt(self.reward_rms.var)),
            }
        )
        if cfg.verbose:
            print(f"[RND] update {self.update_count}: mse={mean_loss:.6g}")
        return np.asarray(losses, dtype=np.float32) if return_losses else mean_loss

    def update_from_buffer(
        self,
        buffer: Any,
        key: str = "next_obs",
        fallback_keys: Sequence[str] = ("obs",),
        **kwargs,
    ) -> float:
        """Fit ``f_hat`` on the states stored in a :class:`RolloutBuffer`.

        By default the **next** states are used, consistently with the paper's
        bonus definition.  Accepts a ``RolloutBuffer``, a mapping of arrays (as
        returned by ``RolloutBuffer.as_arrays()``) or an iterable of dicts.
        """
        states = None
        if hasattr(buffer, "as_arrays"):
            buffer = buffer.as_arrays()
        if isinstance(buffer, Mapping):
            for k in (key, *fallback_keys):
                if k in buffer and buffer[k] is not None:
                    states = np.asarray(buffer[k])
                    break
        else:
            collected = []
            for item in buffer:  # iterable of transitions/dicts
                if isinstance(item, Mapping):
                    for k in (key, *fallback_keys):
                        if k in item:
                            collected.append(np.asarray(item[k], dtype=np.float32))
                            break
            if collected:
                states = np.asarray(collected)
        if states is None:
            raise ValueError(
                f"Could not locate states to fit the RND predictor (tried '{key}'/'{list(fallback_keys)}')."
            )
        return float(self.update(states, **kwargs))

    # ------------------------------------------------------------- inference
    @torch.no_grad()
    def error_of(self, obs: Any) -> float:
        """Single-state raw squared error (convenience for tests)."""
        return float(self.squared_error(obs)[0])

    @property
    def mean_error(self) -> float:
        """Mean raw squared error observed for the last processed batch."""
        return float(self._last_mean_error)

    @property
    def mean_bonus(self) -> float:
        """Mean normalized bonus produced for the last processed batch."""
        return float(self._last_mean_bonus)

    # ------------------------------------------------------------ serialization
    def state_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target.state_dict(),
            "predictor": self.predictor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "obs_rms": self.obs_rms.state_dict(),
            "reward_rms": self.reward_rms.state_dict(),
            "obs_dim": self.obs_dim,
            "update_count": self.update_count,
            "config": {
                "net_arch": tuple(self.config.net_arch),
                "output_dim": self.config.output_dim,
                "activation": self.config.activation,
                "coef": self.config.coef,
            },
        }

    def load_state_dict(self, state: Mapping[str, Any], load_optimizer: bool = True) -> "RND":
        self.target.load_state_dict(state["target"])
        self.predictor.load_state_dict(state["predictor"])
        if load_optimizer and state.get("optimizer") is not None:
            try:
                self.optimizer.load_state_dict(state["optimizer"])
            except Exception:  # architecture mismatch -> keep fresh optimizer
                pass
        if state.get("obs_rms") is not None:
            self.obs_rms.load_state_dict(state["obs_rms"])
        if state.get("reward_rms") is not None:
            self.reward_rms.load_state_dict(state["reward_rms"])
        self.update_count = int(state.get("update_count", self.update_count))
        return self

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)
        return path

    def load(self, path: str, map_location: Optional[str] = None) -> "RND":
        state = torch.load(path, map_location=map_location or self.device)
        return self.load_state_dict(state)

    # ---------------------------------------------------------------- utility
    def reset_reward_statistics(self) -> None:
        """Restart the bonus normalizer (used when a new refining phase begins)."""
        self.reward_rms = RunningMeanStd(shape=(), clip=None)

    def decay_report(self, first: int = 10, last: int = 10) -> Dict[str, float]:
        """Evidence for the paper's claim that bonuses decay with coverage.

        Compares the mean raw error of the earliest updates with the most recent
        ones (``nan`` when not enough history is available).
        """
        errs = [h["mean_error"] for h in self.history if np.isfinite(h["mean_error"])]
        if len(errs) < max(1, first + last):
            return {"early_error": float("nan"), "late_error": float("nan"), "ratio": float("nan")}
        early = float(np.mean(errs[:first]))
        late = float(np.mean(errs[-last:]))
        ratio = float(late / early) if early > 0 else float("nan")
        return {"early_error": early, "late_error": late, "ratio": ratio}

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"RND(obs_dim={self.obs_dim}, output_dim={self.config.output_dim}, "
            f"net_arch={tuple(self.config.net_arch)}, coef={self.config.coef}, "
            f"updates={self.update_count}, device={self.device})"
        )


# ---------------------------------------------------------------------------
# Functional helpers
# ---------------------------------------------------------------------------
def compute_rnd_bonus(
    rnd: RND,
    obs: Any,
    normalize: Optional[bool] = None,
    update_stats: bool = True,
) -> np.ndarray:
    """Free-function form of ``rnd.intrinsic_reward(obs)`` (paper's ``R^RND``)."""
    return rnd.intrinsic_reward(obs, normalize=normalize, update_stats=update_stats)


def make_rnd(
    observation_space: Any = None,
    obs_dim: Optional[int] = None,
    config: Optional[Union[RNDConfig, Mapping[str, Any]]] = None,
    **kwargs,
) -> RND:
    """Builder accepting a config dataclass, a mapping (e.g. parsed YAML) or kwargs.

    ``lambda``/``coef``/``lambda_`` keys are all accepted for the intrinsic
    reward coefficient, since the paper's Table 3 names it ``lambda`` while
    Python reserves the identifier.
    """
    if isinstance(config, Mapping):
        cfg_kwargs = {k: v for k, v in config.items() if k not in ("enabled",)}
        if "lambda" in cfg_kwargs:
            cfg_kwargs["coef"] = cfg_kwargs.pop("lambda")
        if "lambda_" in cfg_kwargs:
            cfg_kwargs["coef"] = cfg_kwargs.pop("lambda_")
        cfg_kwargs.update(kwargs)
        return RND(observation_space=observation_space, obs_dim=obs_dim, **cfg_kwargs)

    if "lambda" in kwargs:
        kwargs["coef"] = kwargs.pop("lambda")
    if "lambda_" in kwargs:
        kwargs["coef"] = kwargs.pop("lambda_")
    return RND(observation_space=observation_space, obs_dim=obs_dim, config=config, **kwargs)
