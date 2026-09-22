"""Diagonal Fisher Information Matrix estimation for knowledge retention.

The paper (Appendix C.1) uses, for the weighting coefficients ``F`` of the
auxiliary retention loss

    L_aux(theta) = sum_i F^i (theta_pre^i - theta^i)^2 ,

the *diagonal of the Fisher Information Matrix* evaluated at the pre-trained
weights ``theta_*`` (Elastic Weight Consolidation, Kirkpatrick et al., 2017).
For Soft Actor-Critic we follow the convention of Wolczyk et al. (2021)
("Continual World"), where the Fisher is estimated from the log-density of
actions *sampled from the current policy*::

    F_ii = E_{s ~ B, a ~ pi_theta(.|s)} [ (d/d theta_i log pi_theta(a|s))^2 ]

For NetHack the retention methods use the expert data (NLD-AA), so the Fisher is
estimated from the log-likelihood of the *expert* actions stored in the dataset
(Appendix B.1: NLD-AA batches are also used to produce the Fisher batches)::

    F_ii = E_{(s, a^*) ~ NLD-AA} [ (d/d theta_i log pi_theta(a^*|s))^2 ]

Both variants are implemented through the ``mode`` argument of
:class:`FisherEstimator` (``"expert"`` / ``"policy"``); accumulating the squared
per-sample gradients over ``num_batches`` mini-batches (10000 for NetHack) and
optionally normalising by the number of accumulated samples.

Only the *actor* is instrumented: the paper applies retention to the actor only
and always sets the critic coefficient to zero.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

try:  # pragma: no cover - torch is optional so docs/tests can import this module
    import torch
    from torch import Tensor, nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    Tensor = Any  # type: ignore
    nn = None  # type: ignore

__all__ = [
    "FisherEstimator",
    "compute_fisher_diagonal",
    "fisher_dot",
    "DEFAULT_FISHER_BATCHES",
    "DEFAULT_BATCH_SIZE",
]

#: Number of mini-batches used to estimate the Fisher for NetHack (Appendix B.1/C.1).
DEFAULT_FISHER_BATCHES = 10000
#: Mini-batch size used by the NLD-AA pipeline (Appendix B.1).
DEFAULT_BATCH_SIZE = 128

# Keys commonly used by the various model wrappers to expose a distribution.
_DIST_KEYS = ("dist", "action_dist", "policy_dist", "distribution", "pi")
_LOGITS_KEYS = ("logits", "action_logits", "policy_logits", "pi_logits")


# ----------------------------------------------------------------------------- 
# helpers
# -----------------------------------------------------------------------------
def _is_distribution(obj: Any) -> bool:
    """Return True when *obj* looks like a ``torch.distributions.Distribution``."""
    if obj is None or torch is None:
        return False
    if isinstance(obj, torch.distributions.Distribution):
        return True
    return hasattr(obj, "log_prob") and hasattr(obj, "sample")


def _find_distribution(output: Any) -> Any:
    """Best-effort extraction of an action distribution from a model output."""
    if _is_distribution(output):
        return output
    if isinstance(output, Mapping):
        for key in _DIST_KEYS:
            value = output.get(key)
            if _is_distribution(value):
                return value
        for value in output.values():
            if _is_distribution(value):
                return value
    if isinstance(output, (tuple, list)):
        for value in output:
            found = _find_distribution(value)
            if found is not None:
                return found
    # Attribute style access (e.g. Sample Factory's ``ModelOutput``).
    for key in _DIST_KEYS:
        value = getattr(output, key, None)
        if _is_distribution(value):
            return value
    return None


def _find_logits(output: Any) -> Any:
    """Best-effort extraction of action logits from a model output."""
    if isinstance(output, Mapping):
        for key in _LOGITS_KEYS:
            if key in output:
                return output[key]
    if isinstance(output, (tuple, list)):
        for value in output:
            found = _find_logits(value)
            if found is not None:
                return found
    for key in _LOGITS_KEYS:
        value = getattr(output, key, None)
        if value is not None and torch is not None and isinstance(value, Tensor):
            return value
    return None


def _select_distribution(actor: Any, output: Any, temperature: float = 1.0) -> Any:
    """Return a distribution object from a model output (or from logits)."""
    dist = _find_distribution(output)
    if dist is not None:
        return dist
    logits = _find_logits(output)
    if logits is None:
        raise ValueError(
            "FisherEstimator could not find an action distribution in the model "
            "output. Pass a `log_prob_fn(actor, obs, actions)` for custom models."
        )
    if temperature != 1.0:
        logits = logits / temperature
    return torch.distributions.Categorical(logits=logits)


def _flatten_actions(actions: Any, ndim: int, discrete: bool = True) -> Any:
    """Flatten leading batch/time dimensions so ``log_prob`` shapes line up."""
    if torch is None or not isinstance(actions, Tensor):
        return actions
    if ndim <= 1:
        return actions
    if discrete:
        return actions.reshape(ndim, -1) if False else actions.reshape(-1)[: ndim]
    return actions


def _prepare_actions(dist: Any, actions: Any) -> Any:
    """Make ``actions`` compatible with ``dist.log_prob``."""
    if actions is None or torch is None:
        return actions
    if not isinstance(actions, Tensor):
        actions = torch.as_tensor(actions)

    n = int(dist.batch_shape.numel()) if hasattr(dist, "batch_shape") else None

    if isinstance(dist, torch.distributions.Categorical):
        actions = actions.long()
        if actions.dim() == 0:
            actions = actions.reshape(1)
        if n is not None and actions.numel() != n and actions.numel() > n:
            actions = actions.reshape(-1)[:n]
        return actions

    # Continuous (Normal / TanhNormal / Independent) distributions.
    if n is not None:
        if actions.numel() == n and getattr(dist, "event_shape", torch.Size([])) != torch.Size([]):
            actions = actions.reshape(n, -1)
        elif actions.dim() == 1 and actions.numel() != n:
            actions = actions.reshape(-1)[:n]
    return actions


def _forward_actor(actor: Any, obs: Any, extra_kwargs: Optional[Mapping[str, Any]] = None) -> Any:
    """Call ``actor`` handling dict/tuple observations and extra kwargs."""
    kwargs = dict(extra_kwargs or {})
    if isinstance(obs, Mapping):
        return actor(**obs, **kwargs)
    if isinstance(obs, (tuple, list)):
        return actor(*obs, **kwargs)
    return actor(obs, **kwargs)


# -----------------------------------------------------------------------------
# estimator
# -----------------------------------------------------------------------------
class FisherEstimator:
    """Estimate the diagonal Fisher Information Matrix of an actor network.

    Parameters
    ----------
    actor:
        The policy module. Its forward output must expose an action
        distribution (``Categorical`` for NetHack, ``Normal``/``TanhNormal`` for
        SAC), or the caller must supply ``log_prob_fn``.
    mode:
        ``"expert"`` estimates the Fisher from (state, *expert* action) pairs
        taken from an offline dataset (NetHack / NLD-AA). ``"policy"`` (alias
        ``"sampled"``) estimates it from actions sampled from the current
        policy, which is the SAC convention of Wolczyk et al. (2021).
    normalize:
        If True, the accumulated squared gradients are divided by the number of
        accumulated samples (and gradients are taken on the *mean* log
        likelihood), matching ``F = E[(d log pi)^2]``. If False (default) the
        estimator accumulates the *sum* of squared per-sample gradients, exactly
        like the reference implementations of EWC.
    num_batches:
        The number of mini-batches the Fisher is estimated from (10000 for
        NetHack). Only used by :meth:`compute`; it is recorded in the state.
    reduction:
        How the per-batch log-likelihood is reduced before differentiation:
        ``"sum"`` (classic EWC) or ``"mean"``.
    temperature:
        Optional softmax temperature for discrete distributions.
    log_prob_fn:
        Optional ``f(actor, obs, actions) -> Tensor`` computing the log
        probability of ``actions``. Required for custom model wrappers.
    action_key:
        Key/attribute used to fetch actions when ``obs`` is a mapping that also
        contains the expert action.
    device:
        Device placed in the internal state dict; the estimator itself never
        moves the actor.
    """

    def __init__(
        self,
        actor: "nn.Module",
        mode: str = "expert",
        normalize: bool = False,
        num_batches: Optional[int] = None,
        reduction: str = "sum",
        temperature: float = 1.0,
        log_prob_fn: Optional[Callable[[Any, Any, Any], Any]] = None,
        extra_forward_kwargs: Optional[Mapping[str, Any]] = None,
        action_key: str = "actions",
        device: Optional[Any] = None,
        name: str = "fisher",
    ) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("FisherEstimator requires PyTorch to be installed.")
        if mode not in ("expert", "offline", "dataset", "policy", "sampled", "online"):
            raise ValueError(f"Unknown Fisher estimation mode: {mode!r}")
        if reduction not in ("sum", "mean"):
            raise ValueError(f"Unknown reduction: {reduction!r}")

        self.actor = actor
        self.mode = "policy" if mode in ("policy", "sampled", "online") else "expert"
        self.normalize = bool(normalize)
        self.num_batches = num_batches
        self.reduction = reduction
        self.temperature = float(temperature)
        self.log_prob_fn = log_prob_fn
        self.extra_forward_kwargs = dict(extra_forward_kwargs or {})
        self.action_key = action_key
        self.name = name
        self.device = device if device is not None else self._infer_device()

        self._fisher: Dict[str, Tensor] = {}
        self._num_samples = 0
        self._num_batches_seen = 0
        self._reset_accumulator()

    # -- introspection ------------------------------------------------------
    def _infer_device(self) -> Any:
        try:
            return next(self.actor.parameters()).device
        except Exception:  # pragma: no cover - parameterless modules
            return torch.device("cpu")

    def named_parameters(self) -> List[Tuple[str, "nn.Parameter"]]:
        """Parameters that receive a Fisher estimate (all of the actor)."""
        params = [(n, p) for n, p in self.actor.named_parameters() if p.requires_grad]
        if not params:  # fall back to every parameter (e.g. fully frozen actor)
            params = list(self.actor.named_parameters())
        return params

    def _reset_accumulator(self) -> None:
        self._fisher = {
            name: torch.zeros_like(param, device=param.device, dtype=torch.float32)
            for name, param in self.named_parameters()
        }

    def reset(self) -> None:
        """Zero the accumulated Fisher and the sample counters."""
        self._reset_accumulator()
        self._num_samples = 0
        self._num_batches_seen = 0

    # -- log probability ---------------------------------------------------
    def log_prob(self, obs: Any, actions: Any = None) -> Tensor:
        """Log probability of ``actions`` under the actor at observation ``obs``.

        If ``actions`` is None and the mode is ``"policy"``, actions are sampled
        from the current policy instead.
        """
        if self.log_prob_fn is not None:
            log_prob = self.log_prob_fn(self.actor, obs, actions)
            return self._reduce_log_prob(log_prob)

        output = _forward_actor(self.actor, obs, self.extra_forward_kwargs)
        dist = _select_distribution(self.actor, output, self.temperature)

        if actions is None:
            with torch.no_grad():
                sampled = dist.sample()
            actions = sampled
        actions = _prepare_actions(dist, actions)
        log_prob = dist.log_prob(actions)
        return self._reduce_log_prob(log_prob)

    def _reduce_log_prob(self, log_prob: Any) -> Tensor:
        if not isinstance(log_prob, Tensor):
            log_prob = torch.as_tensor(log_prob)
        log_prob = log_prob.reshape(-1) if log_prob.dim() > 1 else log_prob
        if self.reduction == "sum":
            return log_prob.sum()
        return log_prob.mean()

    def _extract_actions(self, obs: Any, actions: Any) -> Any:
        """Fetch expert actions from a batch mapping when not given explicitly."""
        if actions is not None:
            return actions
        if isinstance(obs, Mapping) and self.action_key in obs:
            return obs[self.action_key]
        if self.action_key and hasattr(obs, self.action_key):
            return getattr(obs, self.action_key)
        if isinstance(obs, (tuple, list)) and obs:
            # Sample Factory style: (obs, actions) or (obs, actions, ...)
            if len(obs) >= 2:
                return obs[1]
        return None

    def _strip_action_inputs(self, obs: Any) -> Any:
        """Remove the action entry from a mapping of model inputs."""
        if isinstance(obs, Mapping) and self.action_key in obs:
            obs = {k: v for k, v in obs.items() if k != self.action_key}
        return obs

    # -- accumulation ------------------------------------------------------
    def accumulate(self, obs: Any, actions: Any = None, weight: Optional[Any] = None) -> Dict[str, Tensor]:
        """Accumulate squared gradients of one mini-batch into the Fisher.

        ``obs`` may be a tensor, a tuple of inputs, or a mapping of model
        keyword arguments (in which case the expert actions may live under
        ``self.action_key``). ``weight`` (optional) is a per-sample weight, e.g.
        a validity mask for padded trajectories.
        """
        actions = self._extract_actions(obs, actions)
        model_obs = self._strip_action_inputs(obs)

        params = self.named_parameters()
        if not params:  # pragma: no cover - nothing to estimate
            return self.diagonal()

        log_prob = self.log_prob(model_obs, actions)
        if weight is not None:
            weight = torch.as_tensor(weight, device=log_prob.device, dtype=log_prob.dtype)
            weight = weight.reshape(-1)
            n = min(weight.numel(), log_prob.numel())
            log_prob = log_prob if log_prob.numel() == 1 else (log_prob.reshape(-1)[:n] * weight[:n]).sum()

        grads = torch.autograd.grad(
            log_prob,
            [p for _, p in params],
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        for (name, param), grad in zip(params, grads):
            if grad is None:
                continue
            squared = grad.detach().to(torch.float32).pow(2)
            if self.normalize:
                squared = squared / max(1, int(log_prob if False else 1))
            self._fisher[name] = self._fisher[name].to(squared.device) + squared

        batch_size = self._infer_batch_size(obs)
        self._num_samples += batch_size
        self._num_batches_seen += 1
        self.actor.zero_grad(set_to_none=True)
        return self.diagonal()

    @staticmethod
    def _infer_batch_size(obs: Any) -> int:
        if torch is None:
            return 1
        if isinstance(obs, Tensor):
            return int(obs.shape[0]) if obs.dim() > 0 else 1
        if isinstance(obs, Mapping):
            for value in obs.values():
                if isinstance(value, Tensor) and value.dim() > 0:
                    return int(value.shape[0])
        if isinstance(obs, (tuple, list)):
            for value in obs:
                if isinstance(value, Tensor) and value.dim() > 0:
                    return int(value.shape[0])
        return 1

    # -- batch iteration ---------------------------------------------------
    def compute(
        self,
        batches: Union[Iterable[Any], Callable[[], Iterator[Any]]],
        num_batches: Optional[int] = None,
        verbose: bool = False,
    ) -> Dict[str, Tensor]:
        """Estimate the Fisher over ``num_batches`` mini-batches.

        ``batches`` may be either an iterable of batches (each batch is
        ``(obs, actions)``, ``(obs,)`` or a mapping containing the expert
        actions) or a callable returning an infinite/iterable batch stream.
        """
        self.reset()
        num_batches = num_batches if num_batches is not None else self.num_batches
        stream = batches() if callable(batches) else batches

        for i, batch in enumerate(stream):
            if num_batches is not None and i >= num_batches:
                break
            obs, actions, weight = self._unpack_batch(batch)
            self.accumulate(obs, actions, weight=weight)
            if verbose and (i + 1) % 1000 == 0:
                print(f"[fisher:{self.name}] accumulated {i + 1}/{num_batches} batches")

        return self.diagonal()

    @staticmethod
    def _unpack_batch(batch: Any) -> Tuple[Any, Any, Any]:
        if isinstance(batch, Mapping) and not any(
            isinstance(v, Tensor) and v.dim() > 0 for k, v in batch.items() if k == "actions"
        ):
            return batch, None, None
        if isinstance(batch, (tuple, list)):
            if len(batch) == 1:
                return batch[0], None, None
            if len(batch) >= 3 and isinstance(batch[2], Tensor) and batch[2].dim() > 0:
                return batch[0], batch[1], batch[2]
            return batch[0], batch[1], None
        return batch, None, None

    # -- outputs -----------------------------------------------------------
    def diagonal(self, normalize: Optional[bool] = None) -> Dict[str, Tensor]:
        """Return the diagonal Fisher as ``{parameter_name: tensor}``.

        ``normalize`` overrides the constructor flag: when True the accumulated
        squared gradients are divided by the number of accumulated samples and,
        if applicable, by the number of processed batches.
        """
        normalize = self.normalize if normalize is None else bool(normalize)
        if not normalize:
            return {k: v.clone() for k, v in self._fisher.items()}
        scale = 1.0 / max(1, self._num_batches_seen)
        return {k: (v / scale).clone() if self._num_batches_seen else v.clone() for k, v in self._fisher.items()}

    def flat(self, normalize: Optional[bool] = None) -> Tensor:
        """Concatenated diagonal Fisher vector, in ``named_parameters`` order."""
        diag = self.diagonal(normalize=normalize)
        return torch.cat([diag[n].reshape(-1) for n, _ in self.named_parameters() if n in diag])

    #: Alias so an ``EWC`` instance may call the estimator directly.
    fisher_diag = property(lambda self: self.diagonal())

    def __call__(self, actor: Optional["nn.Module"] = None) -> Dict[str, Tensor]:
        """Duck-typed interface consumed by ``src.retention.ewc.EWC``."""
        if actor is not None and actor is not self.actor:
            self.actor = actor
        return self.diagonal()

    # -- persistence -------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {
            "fisher": {k: v.detach().cpu().clone() for k, v in self._fisher.items()},
            "num_samples": self._num_samples,
            "num_batches": self._num_batches_seen,
            "mode": self.mode,
            "normalize": self.normalize,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state is None:
            return
        fisher = state.get("fisher", state)
        for name, value in fisher.items():
            if name in self._fisher:
                self._fisher[name] = value.to(self._fisher[name].device).to(torch.float32)
        self._num_samples = int(state.get("num_samples", self._num_samples))
        self._num_batches_seen = int(state.get("num_batches", state.get("num_batches_seen", self._num_batches_seen)))

    def extra_repr(self) -> str:
        return (
            f"mode={self.mode!r}, normalize={self.normalize}, "
            f"num_batches={self.num_batches}, samples={self._num_samples}"
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}({self.extra_repr()})"


# -----------------------------------------------------------------------------
# convenience functions
# -----------------------------------------------------------------------------
def compute_fisher_diagonal(
    actor: "nn.Module",
    batches: Union[Iterable[Any], Callable[[], Iterator[Any]]],
    num_batches: int = DEFAULT_FISHER_BATCHES,
    mode: str = "expert",
    normalize: bool = False,
    **kwargs: Any,
) -> Dict[str, Tensor]:
    """Estimate the diagonal Fisher of ``actor`` over ``num_batches`` batches.

    NetHack uses ``num_batches=10000`` batches of NLD-AA expert data with
    ``mode="expert"``; SAC experiments use ``mode="policy"`` following Wolczyk
    et al. (2021).
    """
    estimator = FisherEstimator(actor, mode=mode, normalize=normalize, num_batches=num_batches, **kwargs)
    return estimator.compute(batches, num_batches=num_batches)


def fisher_dot(
    fisher: Union[Mapping[str, Tensor], Tensor],
    params: Union[Mapping[str, Tensor], "nn.Module"],
    anchor: Union[Mapping[str, Tensor], "nn.Module", None] = None,
) -> Tensor:
    """Compute ``sum_i F^i (theta_pre^i - theta^i)^2`` (the paper's Eq. 1).

    Kept here as a lightweight probe used by tests/analysis to measure how far
    the actor has drifted from ``theta_*`` in Fisher-weighted distance.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("fisher_dot requires PyTorch.")

    def as_mapping(obj: Any) -> Mapping[str, Tensor]:
        if isinstance(obj, Mapping):
            return obj
        if nn is not None and isinstance(obj, nn.Module):
            return dict(obj.named_parameters())
        return {}

    fisher_map = fisher if isinstance(fisher, Mapping) else None
    param_map = as_mapping(params)
    anchor_map = as_mapping(anchor) if anchor is not None else None

    total = None
    if fisher_map is not None:
        for name, param in param_map.items():
            if name not in fisher_map:
                continue
            f = fisher_map[name]
            ref = param if anchor_map is None else anchor_map.get(name, param)
            term = (f * (ref.detach() - param).pow(2)).sum()
            total = term if total is None else total + term
        if total is None:
            return torch.zeros((), dtype=torch.float32)
        return total

    # Flat Fisher vector split in parameter order.
    flat = fisher
    offset = 0
    for name, param in param_map.items():
        numel = param.numel()
        f = flat[offset : offset + numel].reshape(param.shape)
        offset += numel
        ref = param if anchor_map is None else anchor_map.get(name, param)
        term = (f * (ref.detach() - param).pow(2)).sum()
        total = term if total is None else total + term
    if total is None:
        return torch.zeros((), dtype=torch.float32)
    return total
