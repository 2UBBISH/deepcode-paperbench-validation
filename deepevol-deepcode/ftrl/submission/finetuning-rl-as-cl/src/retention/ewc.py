"""Elastic Weight Consolidation (EWC) retention loss.

Paper: "Fine-tuning Reinforcement Learning Models is Secretly a Forgetting
Mitigation Problem".

Section 2 (Knowledge retention):

    EWC is a regularization-based approach that applies a penalty on parameter
    changes by introducing an auxiliary loss

        L_aux(theta) = sum_i F^i (theta_pre^i - theta^i)^2,                (1)

    where theta (resp. theta_pre) are the weights of the current (resp.
    pre-trained) model, and F is the diagonal of the Fisher matrix.

Appendix C.1 repeats equation (1) verbatim and states that in EWC
(Kirkpatrick et al., 2017), F is the diagonal of the Fisher Information Matrix
(see Wolczyk et al., 2021 for its implementation in Soft Actor-Critic).

Appendix B.3 / the paper's experimental protocol: *knowledge retention is
applied to the actor only*, the critic coefficient is always 0.  Coefficients
used in the paper:

    * NetHack (Human Monk, APPO)  : ewc_coef = 2e6  (Appendix B.1 / B.3)
    * Meta-World RoboticSequence  : ewc_coef = 100  (Appendix B.3, SAC actor)

This module exposes

    * :class:`EWC`                     -- stateful helper attached to an actor
    * :func:`ewc_loss`                 -- functional form of equation (1)
    * :func:`diagonal_fisher_penalty`  -- same, from a ``{name: tensor}`` dict

The penalty is ``0`` when ``theta == theta_pre`` (up to numerical noise), which
is asserted by ``tests/test_retention_losses.py``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Union

try:  # torch is a hard requirement at training time, but keep imports defensive
    import torch
    from torch import Tensor
    from torch.nn import Module
except Exception:  # pragma: no cover - allows importing docs/tests w/o torch
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    Module = Any  # type: ignore[misc,assignment]

__all__ = [
    "EWC",
    "ewc_loss",
    "diagonal_fisher_penalty",
    "normalize_param_name",
    "DEFAULT_EWC_COEF",
]

#: Coefficients stated in the paper for the two main environments.
DEFAULT_EWC_COEF: Dict[str, float] = {
    "nethack": 2e6,
    "montezuma": 1.0,
    "robotic_sequence": 100.0,
}

_TRAILING_PRE_SUFFIXES = ("_pre", "_pretrained", "_anchor", "_prev", "_star")
_LEADING_PREFIXES = ("actor.", "policy.", "module.", "model.", "net.")


def normalize_param_name(name: str) -> str:
    """Canonicalise a parameter name for Fisher/parameter matching.

    Sample-factory style checkpoints rename pre-trained parameters by appending a
    ``_pre``-like suffix and/or wrap the actor in ``actor.`` / ``module.``
    prefixes.  Stripping both sides lets the Fisher matrix computed on the
    pre-trained checkpoint be matched against the fine-tuned actor.
    """
    canonical = name
    for prefix in _LEADING_PREFIXES:
        if canonical.startswith(prefix):
            canonical = canonical[len(prefix) :]
    changed = True
    while changed:
        changed = False
        for suffix in _TRAILING_PRE_SUFFIXES:
            if canonical.endswith(suffix):
                canonical = canonical[: -len(suffix)]
                changed = True
    return canonical


def _as_flat_map(obj: Union[Tensor, Mapping[str, Tensor]]) -> Dict[str, Tensor]:
    """Normalise a Fisher representation into a ``{name: tensor}`` mapping."""
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        return {normalize_param_name(k): v for k, v in obj.items()}
    return {"__flat__": obj}


def ewc_loss(
    params: Mapping[str, Tensor],
    anchor: Mapping[str, Tensor],
    fisher: Optional[Mapping[str, Tensor]] = None,
    coef: float = 1.0,
) -> Tensor:
    """Equation (1): ``L_aux(theta) = sum_i F^i (theta_pre^i - theta^i)^2``.

    Args:
        params: current ``{name: tensor}`` actor parameters (theta).
        anchor: pre-trained ``{name: tensor}`` actor parameters (theta_pre).
        fisher: diagonal Fisher ``{name: tensor}`` (F^i).  ``None`` degenerates
            to uniform weighting ``F^i = 1`` (a plain L2 anchoring penalty).
        coef: scalar multiplier applied to the whole penalty. ``0`` disables it
            (used for the critic, for which the paper always uses coefficient 0).

    Returns:
        Scalar tensor. Differentiable with respect to ``params``.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required for the EWC loss.")

    total = None
    for name, theta in params.items():
        key = normalize_param_name(name)
        if key not in anchor:
            continue
        theta_pre = anchor[key]
        if fisher is None:
            f = None
        else:
            f = fisher.get(key, fisher.get(name))
            if f is None:
                f = fisher.get("__flat__")
        term = (theta_pre - theta) ** 2
        if f is not None:
            term = term * f
        term = term.sum()
        total = term if total is None else total + term

    if total is None:
        return torch.zeros((), dtype=torch.float32)
    return float(coef) * total


def diagonal_fisher_penalty(
    module: Module,
    fisher: Union[Tensor, Mapping[str, Tensor]],
    anchor: Optional[Mapping[str, Tensor]] = None,
    coef: float = 1.0,
) -> Tensor:
    """Compute equation (1) for a live ``torch.nn.Module``.

    ``fisher`` may be a flat tensor (concatenated over ``module.parameters()``
    in order) or a ``{name: tensor}`` mapping.  When ``anchor`` is ``None`` the
    current parameters are snapshotted on the fly, which yields exactly zero.
    """
    params = {normalize_param_name(n): p for n, p in module.named_parameters()}
    if anchor is None:
        anchor = {k: v.detach().clone() for k, v in params.items()}
    fisher_map = _as_flat_map(fisher)
    return ewc_loss(params, anchor, fisher_map, coef=coef)


class EWC:
    """Elastic Weight Consolidation regulariser (actor only).

    The class stores the pre-trained weights ``theta_pre`` (the *anchor*) and the
    diagonal Fisher matrix ``F`` of the actor at ``theta_pre``, then adds
    equation (1) to the RL objective::

        total_loss = rl_loss + ewc.coef * ewc.penalty()

    Usage::

        ewc = EWC(actor, fisher_diag=fisher, coef=2e6)   # NetHack
        ...
        loss = rl_loss + ewc.penalty_loss()              # actor only!

    Parameters
    ----------
    actor:
        The policy network being fine-tuned (``torch.nn.Module``).
    fisher_diag:
        Diagonal Fisher. Either a tensor / ``{name: tensor}`` mapping, or a
        callable ``fisher(actor) -> tensor|dict`` (e.g. built by
        :class:`src.retention.fisher.FisherEstimator`).  ``None`` means uniform
        weighting.
    coef:
        Scalar coefficient for the penalty (NetHack ``2e6``, Meta-World ``100``).
    normalize:
        If ``True`` (default), divide the penalty by the number of Fisher
        elements so the coefficient is scale-independent w.r.t. network size.
        The paper's coefficients assume the un-normalised sum over all elements,
        so the default is ``False``.
    anchor:
        Optional pre-computed anchor ``{name: tensor}``.  If not given it is
        snapshotted from ``actor`` at construction time.
    """

    def __init__(
        self,
        actor: Module,
        fisher_diag: Optional[
            Union[Tensor, Mapping[str, Tensor], Callable[[Module], Any]]
        ] = None,
        coef: float = 1.0,
        normalize: bool = False,
        anchor: Optional[Mapping[str, Tensor]] = None,
        name: str = "ewc",
    ) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required to construct EWC.")
        self.name = name
        self.coef = float(coef)
        self.normalize = bool(normalize)

        self.actor = actor
        self.anchor: Dict[str, Tensor] = (
            {normalize_param_name(k): v.detach().clone() for k, v in anchor.items()}
            if anchor is not None
            else self.snapshot(actor)
        )
        self.fisher: Optional[Dict[str, Tensor]] = self._resolve_fisher(fisher_diag)
        self._numel = self._count_fisher_elements()

    # ------------------------------------------------------------------ setup
    @staticmethod
    def snapshot(actor: Module) -> Dict[str, Tensor]:
        """Detached copy of all parameters: ``theta_pre``."""
        return {
            normalize_param_name(n): p.detach().clone()
            for n, p in actor.named_parameters()
        }

    def _resolve_fisher(
        self,
        fisher_diag: Optional[
            Union[Tensor, Mapping[str, Tensor], Callable[[Module], Any]]
        ],
    ) -> Optional[Dict[str, Tensor]]:
        if fisher_diag is None:
            return None
        if callable(fisher_diag) and not isinstance(fisher_diag, Mapping):
            fisher_diag = fisher_diag(self.actor)
        if isinstance(fisher_diag, Mapping):
            return {
                normalize_param_name(k): v.detach().clone()
                if isinstance(v, Tensor)
                else torch.as_tensor(v)
                for k, v in fisher_diag.items()
            }
        if isinstance(fisher_diag, Tensor):
            return self._split_flat_fisher(fisher_diag)
        return None

    def _split_flat_fisher(self, flat: Tensor) -> Dict[str, Tensor]:
        """Split a flat Fisher tensor following ``actor.parameters()`` order."""
        out: Dict[str, Tensor] = {}
        offset = 0
        for name, p in self.actor.named_parameters():
            n = p.numel()
            out[normalize_param_name(name)] = flat[offset : offset + n].view_as(p)
            offset += n
        return out

    def _count_fisher_elements(self) -> int:
        if not self.fisher:
            return max(1, sum(p.numel() for p in self.actor.parameters()))
        total = 0
        for v in self.fisher.values():
            total += int(v.numel())
        return max(1, total)

    # --------------------------------------------------------------- penalty
    def penalty(self, actor: Optional[Module] = None) -> Tensor:
        """Return ``sum_i F^i (theta_pre^i - theta^i)^2`` (equation 1)."""
        actor = actor if actor is not None else self.actor
        params = {n: p for n, p in actor.named_parameters()}
        loss = ewc_loss(params, self.anchor, self.fisher, coef=1.0)
        if self.normalize:
            loss = loss / float(self._numel)
        return loss

    def penalty_loss(self, actor: Optional[Module] = None) -> Tensor:
        """Scaled penalty: ``coef * L_aux(theta)``."""
        return float(self.coef) * self.penalty(actor)

    # ``loss`` is the name used by the runner when adding the auxiliary term.
    def loss(self, actor: Optional[Module] = None) -> Tensor:
        return self.penalty_loss(actor)

    def __call__(self, actor: Optional[Module] = None) -> Tensor:
        return self.penalty_loss(actor)

    # ------------------------------------------------------------ bookkeeping
    def state_dict(self) -> Dict[str, Any]:
        return {
            "coef": self.coef,
            "normalize": self.normalize,
            "anchor": self.anchor,
            "fisher": self.fisher,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> "EWC":
        self.coef = float(state.get("coef", self.coef))
        self.normalize = bool(state.get("normalize", self.normalize))
        if state.get("anchor") is not None:
            self.anchor = {
                normalize_param_name(k): v for k, v in state["anchor"].items()
            }
        if state.get("fisher") is not None:
            self.fisher = {
                normalize_param_name(k): v for k, v in state["fisher"].items()
            }
        self._numel = self._count_fisher_elements()
        return self

    def configure(self, cfg: Any) -> "EWC":
        """Apply ``coef``/``normalize`` from a config object or mapping."""
        get = cfg.get if isinstance(cfg, Mapping) else lambda k, d=None: getattr(cfg, k, d)
        if get("coef", None) is not None:
            self.coef = float(get("coef"))
        if get("normalize", None) is not None:
            self.normalize = bool(get("normalize"))
        return self

    @property
    def enabled(self) -> bool:
        return self.coef != 0.0 and self.fisher is not None

    def extra_repr(self) -> str:
        return (
            f"coef={self.coef}, normalize={self.normalize}, "
            f"params={len(self.anchor)}, fisher={'yes' if self.fisher else 'no'}"
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"EWC({self.extra_repr()})"


def ewc_coef_for(env_name: str) -> float:
    """Convenience lookup of the paper's coefficient per environment."""
    return float(DEFAULT_EWC_COEF.get(str(env_name).lower(), 1.0))
