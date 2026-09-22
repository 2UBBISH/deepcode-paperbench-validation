"""Learning-rate (inverse regularization) schedules for BaM.

In BaM the parameter ``lambda_t > 0`` is the *inverse* regularization parameter
of the regularized objective

    L^BaM(q) := D_hat_{q_t}(q ; p) + (2 / lambda_t) KL(q_t ; q)          (eq. 3)

so that small ``lambda_t`` means strong regularization (learning rate) and large
``lambda_t`` means weak regularization / aggressive updates.  In the limiting
case ``lambda_t -> 0`` the BaM updates have no effect at all
(``Sigma_{t+1} = Sigma_t`` and ``mu_{t+1} = mu_t``), while ``lambda_t -> inf``
(with ``B = 1``) recovers exact Gaussian score matching.

Schedules reported in the paper (Section 5.1, Appendix E.3 and E.4):

* Gaussian targets (constant learning rate):
      lambda_t = B D                                              (Section 5.1)
* Non-Gaussian (sinh-arcsinh) targets and PosteriorDB (decaying rate):
      lambda_t = B D / (t + 1)                                    (Section 5.1)
* Ablated schedules considered in Appendix E.2 / E.4:
      lambda_t = B,  B D,  B / (t + 1),  B D / (t + 1),
      lambda_t = B D / sqrt(t + 1)

Here ``B`` is the batch size and ``D`` the dimension of the latent space.  The
convention throughout is that ``t`` is a 0-based iteration counter, i.e. the
first iteration uses ``lambda_0``.

The module exposes

* :class:`Schedule` -- a picklable wrapper storing the schedule name,
  its parameters and the evaluated values;
* :func:`constant_schedule`, :func:`decay_schedule`, :func:`pow_decay_schedule`,
  :func:`exponential_decay_schedule` -- explicit schedule constructors;
* :func:`bd_schedule`, :func:`b_schedule`, :func:`bd_over_t_schedule`,
  :func:`b_over_t_schedule`, :func:`bd_over_sqrt_t_schedule`;
* :func:`make_schedule` -- the general factory used by :mod:`bam.bam` to
  resolve string names such as ``"BD/(t+1)"``;
* :func:`parse_schedule_name` -- normalizes the many spellings of a name.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Optional, Sequence, Union

import numpy as np

__all__ = [
    "Schedule",
    "constant_schedule",
    "decay_schedule",
    "pow_decay_schedule",
    "exponential_decay_schedule",
    "bd_schedule",
    "b_schedule",
    "bd_over_t_schedule",
    "b_over_t_schedule",
    "bd_over_sqrt_t_schedule",
    "make_schedule",
    "parse_schedule_name",
    "resolve_schedule",
    "schedule_values",
    "SCHEDULE_NAMES",
    "DEFAULT_GAUSSIAN_SCHEDULE",
    "DEFAULT_NON_GAUSSIAN_SCHEDULE",
]

Number = Union[int, float]

#: Name of the constant learning rate used for Gaussian targets (Section 5.1).
DEFAULT_GAUSSIAN_SCHEDULE = "BD"
#: Name of the decaying learning rate used for non-Gaussian / PosteriorDB
#: targets (Sections 5.1-5.2).
DEFAULT_NON_GAUSSIAN_SCHEDULE = "BD/(t+1)"


# --------------------------------------------------------------------------- #
# Name normalization
# --------------------------------------------------------------------------- #
def _normalize(name: str) -> str:
    """Lower-case, strip and remove whitespace/braces from a schedule name."""
    s = str(name).strip().lower()
    for ch in (" ", "\t", "\n", "{", "}", "$", "\\", "_"):
        s = s.replace(ch, "")
    s = s.replace("[", "(").replace("]", ")")
    return s


#: Canonical name -> list of accepted aliases.
_SCHEDULE_ALIASES: dict = {
    "constant": (
        "constant",
        "const",
        "fixed",
        "1",
        "one",
        "constant1",
    ),
    "B": (
        "b",
        "batchsize",
        "b,",  # be permissive
    ),
    "BD": (
        "bd",
        "db",
        "batchsizedim",
        "bdim",
        "b*d",
    ),
    "1/(t+1)": (
        "1/(t+1)",
        "1/(1+t)",
        "inv(t+1)",
        "(t+1)^{-1}",
        "(1+t)^{-1}",
    ),
    "B/(t+1)": (
        "b/(t+1)",
        "b/(1+t)",
        "bover(t+1)",
        "b/(t1)",
    ),
    "BD/(t+1)": (
        "bd/(t+1)",
        "bd/(1+t)",
        "bdover(t+1)",
        "bd/(t1)",
    ),
    "BD/sqrt(t+1)": (
        "bd/sqrt(t+1)",
        "bd/sqrt(1+t)",
        "bd/sqrt(t)",
        "bd/(t+1)^0.5",
        "bd/(t+1)^{0.5}",
        "bd/(t+1)^{1/2}",
        "bd/(1+t)^0.5",
    ),
    "B/sqrt(t+1)": (
        "b/sqrt(t+1)",
        "b/sqrt(1+t)",
        "b/(t+1)^0.5",
        "b/(t+1)^{0.5}",
    ),
}

_NAME_TO_CANONICAL: dict = {}
for _canon, _aliases in _SCHEDULE_ALIASES.items():
    _NAME_TO_CANONICAL[_normalize(_canon)] = _canon
    for _a in _aliases:
        _NAME_TO_CANONICAL[_normalize(_a)] = _canon

#: Canonical schedule names understood by :func:`make_schedule`.
SCHEDULE_NAMES = tuple(sorted(_SCHEDULE_ALIASES.keys()))

_NUMERIC_NAMES = {"constant", "B", "BD"}
_DECAY_POWERS = {
    "1/(t+1)": 1.0,
    "B/(t+1)": 1.0,
    "BD/(t+1)": 1.0,
    "B/sqrt(t+1)": 0.5,
    "BD/sqrt(t+1)": 0.5,
}


def parse_schedule_name(name: Union[str, Number, Callable, "Schedule"]) -> str:
    """Return the canonical schedule name for ``name``.

    Accepts the many spellings used in the paper/plan (``"BD/(t+1)"``,
    ``"b d / (1+t)"``, ``"BD/sqrt(t+1)"``, ``"constant"``, ...) as well as the
    special markers ``"<callable>"`` / ``"<custom>"`` for user callables.

    Raises ``ValueError`` if the name cannot be understood.
    """
    if isinstance(name, Schedule):
        return name.name
    if callable(name):
        return "<callable>"
    if isinstance(name, (int, float, np.integer, np.floating)):
        return "<constant:%g>" % float(name)
    key = _normalize(name)
    if key in _NAME_TO_CANONICAL:
        return _NAME_TO_CANONICAL[key]
    raise ValueError(
        "unknown learning-rate schedule %r; known schedules are %s"
        % (name, ", ".join(SCHEDULE_NAMES))
    )


# --------------------------------------------------------------------------- #
# Schedule object
# --------------------------------------------------------------------------- #
class Schedule:
    """A callable ``t -> lambda_t`` with bookkeeping.

    Parameters
    ----------
    fn:
        Callable mapping a (0-based) iteration index to ``lambda_t``.
    name:
        Human readable / canonical schedule name.
    scale:
        Multiplicative prefactor (e.g. ``B * D`` for the ``BD`` schedules).
    power:
        Exponent of the ``(t + 1)`` decay (``0`` for constant schedules).
    batch_size, dim:
        Batch size ``B`` and dimension ``D`` used to build the schedule.
    params:
        Any extra keyword arguments (kept for reproducibility).
    """

    __slots__ = (
        "fn",
        "name",
        "scale",
        "power",
        "batch_size",
        "dim",
        "params",
    )

    def __init__(
        self,
        fn: Callable[[int], float],
        name: str = "<callable>",
        scale: float = 1.0,
        power: float = 0.0,
        batch_size: Optional[int] = None,
        dim: Optional[int] = None,
        **params: Any,
    ) -> None:
        self.fn = fn
        self.name = name
        self.scale = float(scale)
        self.power = float(power)
        self.batch_size = None if batch_size is None else int(batch_size)
        self.dim = None if dim is None else int(dim)
        self.params = dict(params)

    # -- evaluation ------------------------------------------------------ #
    def __call__(self, t: Any = 0) -> float:
        t = 0 if t is None else t
        return float(self.fn(int(t) if not _is_array_like(t) else t))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            "Schedule(name=%r, scale=%g, power=%g, B=%s, D=%s)"
            % (self.name, self.scale, self.power, self.batch_size, self.dim)
        )

    # -- convenience ------------------------------------------------------ #
    @property
    def is_constant(self) -> bool:
        return self.power == 0.0

    def values(self, n_iter: int) -> np.ndarray:
        """Return ``[lambda_0, ..., lambda_{n_iter-1}]``."""
        return np.array([self(t) for t in range(int(n_iter))], dtype=float)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "scale": self.scale,
            "power": self.power,
            "batch_size": self.batch_size,
            "dim": self.dim,
            **self.params,
        }


def _is_array_like(x: Any) -> bool:
    return hasattr(x, "shape") and not isinstance(x, (str, bytes))


# --------------------------------------------------------------------------- #
# Explicit constructors
# --------------------------------------------------------------------------- #
def constant_schedule(value: Number, batch_size: Optional[int] = None,
                      dim: Optional[int] = None) -> Schedule:
    """``lambda_t = value`` for every ``t``."""
    v = float(value)
    return Schedule(
        lambda t: v,
        name="constant",
        scale=v,
        power=0.0,
        batch_size=batch_size,
        dim=dim,
        value=v,
    )


def pow_decay_schedule(scale: Number, power: Number = 1.0,
                       batch_size: Optional[int] = None,
                       dim: Optional[int] = None,
                       name: Optional[str] = None) -> Schedule:
    """``lambda_t = scale / (t + 1) ** power`` (``power = 0`` -> constant)."""
    s = float(scale)
    p = float(power)

    def fn(t: int) -> float:
        return s / float(t + 1) ** p if p > 0.0 else s

    return Schedule(
        fn,
        name=name if name is not None else "scale/(t+1)**%g" % p,
        scale=s,
        power=p,
        batch_size=batch_size,
        dim=dim,
    )


def decay_schedule(scale: Number, batch_size: Optional[int] = None,
                   dim: Optional[int] = None,
                   name: Optional[str] = None,
                   **kwargs: Any) -> Schedule:
    """``lambda_t = scale / (t + 1)`` (the paper's decaying schedule)."""
    if "power" in kwargs:
        return pow_decay_schedule(
            scale,
            kwargs.pop("power"),
            batch_size=batch_size,
            dim=dim,
            name=name,
        )
    return pow_decay_schedule(scale, 1.0, batch_size=batch_size, dim=dim,
                              name=name)


def exponential_decay_schedule(scale: Number, decay: Number = 0.5,
                               batch_size: Optional[int] = None,
                               dim: Optional[int] = None) -> Schedule:
    """``lambda_t = scale * decay ** t`` (not used in the paper; utility)."""
    s, d = float(scale), float(decay)
    return Schedule(
        lambda t: s * d ** int(t),
        name="exponential",
        scale=s,
        batch_size=batch_size,
        dim=dim,
        decay=d,
    )


# --------------------------------------------------------------------------- #
# The schedules used in the paper
# --------------------------------------------------------------------------- #
def _require(value: Optional[int], what: str) -> int:
    if value is None:
        raise ValueError(
            "the schedule requires the %s; pass %s=... to make_schedule()"
            % (what, "batch_size" if what == "batch size" else "dim")
        )
    return int(value)


def bd_schedule(batch_size: Optional[int] = None, dim: Optional[int] = None,
                **kw: Any) -> Schedule:
    """Constant ``lambda_t = B D``  (used for Gaussian targets, Section 5.1)."""
    B = _require(batch_size, "batch size")
    D = _require(dim, "dimension")
    return constant_schedule(float(B * D), batch_size=B, dim=D)


def b_schedule(batch_size: Optional[int] = None, dim: Optional[int] = None,
               **kw: Any) -> Schedule:
    """Constant ``lambda_t = B`` (ablated constant schedule, Appendix E.3)."""
    B = _require(batch_size, "batch size")
    return constant_schedule(float(B), batch_size=B, dim=dim)


def bd_over_t_schedule(batch_size: Optional[int] = None,
                       dim: Optional[int] = None,
                       power: Number = 1.0,
                       **kw: Any) -> Schedule:
    """``lambda_t = B D / (t + 1)`` (non-Gaussian / PosteriorDB targets)."""
    B = _require(batch_size, "batch size")
    D = _require(dim, "dimension")
    return pow_decay_schedule(float(B * D), power, batch_size=B, dim=D,
                              name="BD/(t+1)" if float(power) == 1.0
                              else "BD/(t+1)^%g" % float(power))


def b_over_t_schedule(batch_size: Optional[int] = None,
                      dim: Optional[int] = None,
                      power: Number = 1.0,
                      **kw: Any) -> Schedule:
    """``lambda_t = B / (t + 1)`` (ablated decaying schedule, Appendix E.3)."""
    B = _require(batch_size, "batch size")
    return pow_decay_schedule(float(B), power, batch_size=B, dim=dim,
                              name="B/(t+1)" if float(power) == 1.0
                              else "B/(t+1)^%g" % float(power))


def bd_over_sqrt_t_schedule(batch_size: Optional[int] = None,
                            dim: Optional[int] = None,
                            **kw: Any) -> Schedule:
    """``lambda_t = B D / sqrt(t + 1)`` (ablated schedule, Appendix E.4)."""
    B = _require(batch_size, "batch size")
    D = _require(dim, "dimension")
    return pow_decay_schedule(float(B * D), 0.5, batch_size=B, dim=D,
                              name="BD/sqrt(t+1)")


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def make_schedule(schedule: Union[str, Number, Callable, Schedule, Sequence] = None,
                  *,
                  batch_size: Optional[int] = None,
                  dim: Optional[int] = None,
                  B: Optional[int] = None,
                  D: Optional[int] = None,
                  value: Optional[Number] = None,
                  power: Optional[Number] = None,
                  scale: Optional[Number] = None,
                  **kwargs: Any) -> Schedule:
    """Build a :class:`Schedule` from a name, number, callable or sequence.

    Parameters
    ----------
    schedule:
        * ``None`` -- defaults to the constant ``B D`` schedule
          (``batch_size``/``dim`` required).
        * a tree of names, ``"constant"``, ``"B"``, ``"BD"``, ``"B/(t+1)"``,
          ``"BD/(t+1)"``, ``"B/sqrt(t+1)"``, ``"BD/sqrt(t+1)"``,
          ``"1/(t+1)"`` (case/space insensitive);
        * a positive number -> constant schedule with that value;
        * a callable ``t -> lambda_t``;
        * a sequence -> constant schedule with ``schedule[0]`` if length 1,
          otherwise a lookup schedule indexed by ``t`` (clipped to the last
          entry), which is handy for passing a precomputed schedule.
    batch_size, dim:
        Batch size ``B`` and dimension ``D``.  ``B``/``D`` are accepted as
        aliases.
    value, scale, power:
        Overrides for the constant value / multiplicative scale / decay power.

    Returns
    -------
    Schedule
    """
    B = batch_size if batch_size is not None else B
    D = dim if dim is not None else D

    # --- callables / Schedule objects ---------------------------------- #
    if isinstance(schedule, Schedule):
        return schedule
    if callable(schedule):
        return Schedule(schedule, name="<callable>", batch_size=B, dim=D)

    # --- sequences ------------------------------------------------------ #
    if isinstance(schedule, (list, tuple, np.ndarray)):
        seq = np.asarray(schedule, dtype=float).ravel()
        if seq.size == 0:
            raise ValueError("cannot build a schedule from an empty sequence")
        if seq.size == 1:
            return constant_schedule(float(seq[0]), batch_size=B, dim=D)

        def fn(t: int, _seq: np.ndarray = seq) -> float:
            i = int(t)
            i = 0 if i < 0 else (i if i < _seq.size else _seq.size - 1)
            return float(_seq[i])

        return Schedule(fn, name="sequence", batch_size=B, dim=D,
                        length=int(seq.size))

    # --- plain numbers --------------------------------------------------- #
    if isinstance(schedule, (int, float, np.integer, np.floating)):
        return constant_schedule(float(schedule), batch_size=B, dim=D)

    # ``None`` falls through to the default below.
    name = parse_schedule_name(schedule) if schedule is not None else "BD"

    if name == "<callable>":  # pragma: no cover - handled above
        raise ValueError("internal error: callable reached string branch")

    if name.startswith("<constant:"):
        return constant_schedule(float(name.split(":", 1)[1].rstrip(">")),
                                 batch_size=B, dim=D)

    # --- named schedules -------------------------------------------------- #
    if name == "constant":
        v = value if value is not None else (scale if scale is not None else 1.0)
        return constant_schedule(v, batch_size=B, dim=D)
    if name == "B":
        return b_schedule(batch_size=B, dim=D)
    if name == "BD":
        return bd_schedule(batch_size=B, dim=D)
    if name == "1/(t+1)":
        p = 1.0 if power is None else float(power)
        return pow_decay_schedule(1.0, p, batch_size=B, dim=D,
                                  name="1/(t+1)")
    if name == "B/(t+1)":
        return b_over_t_schedule(batch_size=B, dim=D,
                                 power=1.0 if power is None else power)
    if name == "BD/(t+1)":
        return bd_over_t_schedule(batch_size=B, dim=D,
                                  power=1.0 if power is None else power)
    if name == "B/sqrt(t+1)":
        return pow_decay_schedule(float(_require(B, "batch size")), 0.5,
                                  batch_size=B, dim=D, name="B/sqrt(t+1)")
    if name == "BD/sqrt(t+1)":
        return bd_over_sqrt_t_schedule(batch_size=B, dim=D)

    raise ValueError(  # pragma: no cover - parse_schedule_name already raised
        "unknown learning-rate schedule %r" % (schedule,)
    )


#: ``make_schedule`` under a more explicit name.
resolve_schedule = make_schedule


def schedule_values(schedule: Union[str, Number, Callable, Schedule, Sequence],
                    n_iter: int,
                    batch_size: Optional[int] = None,
                    dim: Optional[int] = None,
                    **kwargs: Any) -> np.ndarray:
    """Convenience: evaluate a schedule for ``n_iter`` iterations."""
    sched = make_schedule(schedule, batch_size=batch_size, dim=dim, **kwargs)
    return sched.values(n_iter)
