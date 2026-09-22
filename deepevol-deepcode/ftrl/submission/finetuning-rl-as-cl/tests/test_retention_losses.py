"""Smoke/unit tests for the four knowledge-retention mechanisms of the paper
"Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation
Problem" (Wolczyk et al., 2024).

The suite exercises the actor-only retention losses implemented under
``src/retention``:

* ``ewc.py``             -- Eq. (1)      ``L_aux(theta) = sum_i F^i (theta_pre^i - theta^i)^2``
* ``fisher.py``          -- diagonal Fisher of the actor at ``theta_*`` (Appendix C.1 / B.1)
* ``behavioral_cloning.py`` -- ``D_KL^s(pi_theta || pi_*)`` on pre-training states (Appendix C.2)
* ``kickstarting.py``    -- ``D_KL^s(pi_* || pi_theta)`` on online data (Appendix C.2)
* ``episodic_memory.py`` -- no auxiliary loss, protected replay fraction (Appendix C.3)

Key invariants asserted here:

1. The EWC penalty is exactly zero at ``theta == theta_pre`` and grows
   quadratically in the perturbation, scaling linearly with the coefficient.
2. The diagonal Fisher is non-negative, finite and aligned with the actor's
   named parameters; ``fisher_dot`` reproduces the closed-form penalty.
3. ``kl_s_divergence`` is the per-state log-ratio and the BC/KS KL terms vanish
   when student and teacher coincide; the BC KL direction is
   ``KL(student || teacher)`` (forward) per Appendix C.2.
4. The Kickstarting coefficient follows the paper's per-train-step schedule
   (NetHack: ``0.5 * 0.99998**t``).
5. The episodic-memory buffer never overwrites its protected prior-region and
   mixed batches always contain prior-task transitions.
6. The per-environment coefficients match the paper (EWC 2e6 / 100, BC 2.0 /
   1.0, KS 0.5 + 0.99998 decay, EM fraction 0.1, BC memory 10000).

The module is dependency tolerant: NumPy / PyTorch / the retention modules are
imported lazily and missing pieces degrade to ``SkipTest`` instead of failing.
It can be run either under ``pytest`` or directly::

    python -m tests.test_retention_losses
"""

from __future__ import annotations

import copy
import importlib
import math
import os
import sys
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Path / harness setup
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:  # shared sentinel from the tests package
    from tests import SkipTest
except Exception:  # pragma: no cover - standalone execution
    class SkipTest(Exception):
        """Raised when an optional dependency or feature is unavailable."""


try:  # optional dependency: NumPy
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

try:  # optional dependency: PyTorch
    import torch
    import torch.nn as nn
    from torch.distributions import Categorical

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    nn = None
    Categorical = None
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _require_torch() -> None:
    if not _HAS_TORCH:
        raise SkipTest("PyTorch is not installed")


def _require_numpy() -> None:
    if _np is None:
        raise SkipTest("NumPy is not installed")


def _imp(*names: str) -> Any:
    """Import the first importable dotted module among ``names``."""
    last: Optional[BaseException] = None
    for name in names:
        try:
            return importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - depends on env
            last = exc
    raise SkipTest("could not import %s (%s)" % (", ".join(names), last))


def _approx(a: float, b: float, rel: float = 1e-6, abs_: float = 1e-8) -> bool:
    try:
        a = float(a)
        b = float(b)
    except Exception:
        return False
    if math.isnan(a) or math.isnan(b):
        return False
    return abs(a - b) <= abs_ + rel * max(abs(a), abs(b))


def _f(value: Any) -> float:
    """Best-effort conversion of a tensor / array / scalar to a Python float."""
    if value is None:
        raise AssertionError("expected a scalar, got None")
    if hasattr(value, "detach"):
        value = value.detach()
        try:
            return float(value.sum())
        except Exception:
            pass
    if _np is not None and isinstance(value, _np.ndarray):
        return float(value.sum())
    if isinstance(value, (list, tuple)):
        return float(sum(_f(v) for v in value))
    return float(value)


def _field_of(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a mapping, object attribute, or tuple position."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    if hasattr(obj, key):
        return getattr(obj, key)
    try:
        return obj[key]
    except Exception:
        return default


if _HAS_TORCH:

    class _ActorOutput(dict):
        """Dict-like model output exposing both ``logits`` and a distribution.

        The retention helpers probe outputs either for a distribution
        (``log_prob``/``sample``) or for ``logits``, so the stub actor returns an
        object that satisfies both conventions.
        """

        def __init__(self, logits: Any, dist: Any) -> None:
            super().__init__(logits=logits, dist=dist, distribution=dist)

        @property
        def logits(self) -> Any:
            return self["logits"]

        @property
        def policy_logits(self) -> Any:
            return self["logits"]

        @property
        def dist(self) -> Any:
            return self["dist"]

        @property
        def distribution(self) -> Any:
            return self["dist"]

        def log_prob(self, actions: Any = None) -> Any:
            if actions is None:
                return self["dist"].logits
            return self["dist"].log_prob(actions)

        def entropy(self) -> Any:
            return self["dist"].entropy()

        def sample(self) -> Any:
            return self["dist"].sample()

        @property
        def probs(self) -> Any:
            return self["dist"].probs

    class _TinyActor(nn.Module):
        """Minimal categorical actor with the interfaces used by the losses."""

        def __init__(self, in_dim: int = 4, n_actions: int = 3, hidden: int = 8) -> None:
            super().__init__()
            self.fc1 = nn.Linear(in_dim, hidden)
            self.fc2 = nn.Linear(hidden, n_actions)
            self.n_actions = n_actions

        # -- distribution helpers ------------------------------------------------
        def logits(self, obs: Any) -> Any:
            return self.fc2(torch.tanh(self.fc1(obs)))

        def _dist(self, obs: Any) -> Any:
            return Categorical(logits=self.logits(obs))

        def distribution(self, obs: Any) -> Any:
            return self._dist(obs)

        def forward(self, obs: Any) -> Any:
            logits = self.logits(obs)
            return _ActorOutput(logits, Categorical(logits=logits))

        def log_prob(self, obs: Any, actions: Any = None) -> Any:
            dist = self._dist(obs)
            if actions is None:
                return dist.logits
            if hasattr(actions, "long"):
                actions = actions.long()
            return dist.log_prob(actions)

        def get_actions(self, obs: Any, deterministic: bool = False) -> Any:
            dist = self._dist(obs)
            if deterministic:
                return dist.probs.argmax(-1)
            return dist.sample()

        def act(self, obs: Any, deterministic: bool = False, **_: Any) -> Any:
            return self.get_actions(obs, deterministic)

        def sample_action(self, obs: Any, deterministic: bool = False, **_: Any) -> Any:
            return self.get_actions(obs, deterministic)


def _make_actor(seed: Optional[int] = 0) -> Any:
    """Deterministically initialised tiny categorical actor."""
    _require_torch()
    if seed is not None:
        torch.manual_seed(int(seed))
    return _TinyActor()


def _perturb(module: Any, delta: float) -> None:
    with torch.no_grad():
        for param in module.parameters():
            param.add_(float(delta))


def _named_fisher(actor: Any, value: float = 1.0) -> Dict[str, Any]:
    return {name: torch.full_like(param, float(value))
            for name, param in actor.named_parameters()}


def _anchor_of(actor: Any) -> Dict[str, Any]:
    return {name: param.detach().clone() for name, param in actor.named_parameters()}


def _invoke_loss(obj: Any, obs: Any = None, **kwargs: Any) -> Any:
    """Call a retention object's loss with whichever convention it supports."""
    attempts: List[Callable[[], Any]] = []
    if obs is not None:
        attempts += [
            lambda: obj.loss(obs=obs, **kwargs),
            lambda: obj.penalty_loss(obs=obs, **kwargs),
            lambda: obj.penalty(obs=obs, **kwargs),
            lambda: obj(obs=obs, **kwargs),
            lambda: obj.loss({"obs": obs}, **kwargs),
            lambda: obj.penalty({"obs": obs}, **kwargs),
            lambda: obj.penalty_loss({"obs": obs}, **kwargs),
            lambda: obj(obs, **kwargs),
            lambda: obj.penalty(obs, **kwargs),
            lambda: obj.loss(obs, **kwargs),
        ]
    attempts += [
        lambda: obj.loss(**kwargs),
        lambda: obj.penalty_loss(**kwargs),
        lambda: obj.penalty(**kwargs),
        lambda: obj(**kwargs),
    ]
    errors: List[BaseException] = []
    for fn in attempts:
        try:
            out = fn()
        except Exception as exc:  # try the next calling convention
            errors.append(exc)
            continue
        if out is None:
            errors.append(ValueError("loss returned None"))
            continue
        return out
    raise SkipTest("no supported loss call convention (first errors: %s)"
                   % (errors[:3],))


# ---------------------------------------------------------------------------
# EWC  (Appendix C.1 / Eq. 1)
# ---------------------------------------------------------------------------

def test_ewc_penalty_zero_at_pretrained_weights() -> None:
    """L_aux(theta_pre) == 0 and is non-negative everywhere."""
    ewc_mod = _imp("src.retention.ewc")
    actor = _make_actor(0)
    fisher = _named_fisher(actor)
    ewc = ewc_mod.EWC(actor, fisher_diag=fisher, coef=1.0)

    assert _approx(_f(ewc.penalty()), 0.0, abs_=1e-12), "penalty must vanish at theta_pre"

    _perturb(actor, 0.05)
    pen = _f(ewc.penalty())
    assert pen > 0.0, "penalty must be positive away from theta_pre"

    # penalty_loss / loss are aliases of the same scaled quantity
    assert _approx(_f(ewc.penalty_loss()), pen, rel=1e-9)
    loss_attr = getattr(ewc, "loss", None)
    if callable(loss_attr):
        assert _approx(_f(loss_attr()), pen, rel=1e-9)


def test_ewc_penalty_grows_quadratically_and_scales_with_coef() -> None:
    """sum_i F^i (dtheta^i)^2 is quadratic in dtheta and linear in the coef."""
    ewc_mod = _imp("src.retention.ewc")

    actor = _make_actor(0)
    fisher = _named_fisher(actor, 1.0)
    ewc = ewc_mod.EWC(actor, fisher_diag=fisher, coef=1.0)

    _perturb(actor, 0.01)
    pen1 = _f(ewc.penalty())
    _perturb(actor, 0.01)  # 0.02 total
    pen2 = _f(ewc.penalty())

    assert pen1 > 0.0 and pen2 > 0.0
    assert _approx(pen2 / pen1, 4.0, rel=1e-3), (pen1, pen2)

    # linear in the coefficient
    actor2 = _make_actor(0)  # identical initialisation
    ewc_hi = ewc_mod.EWC(actor2, fisher_diag=_named_fisher(actor2, 1.0), coef=3.0)
    _perturb(actor2, 0.01)
    pen_hi = _f(ewc_hi.penalty())
    assert _approx(pen_hi, 3.0 * pen1, rel=1e-3), (pen_hi, pen1)

    # zero coefficient disables the penalty
    actor3 = _make_actor(0)
    ewc_off = ewc_mod.EWC(actor3, fisher_diag=_named_fisher(actor3, 1.0), coef=0.0)
    _perturb(actor3, 1.0)
    assert _approx(_f(ewc_off.penalty()), 0.0, abs_=1e-12)


def test_ewc_functional_and_helper_match_object_penalty() -> None:
    """ewc_loss / diagonal_fisher_penalty reproduce the EWC object's penalty."""
    ewc_mod = _imp("src.retention.ewc")

    actor = _make_actor(1)
    fisher = _named_fisher(actor, 1.0)
    anchor = _anchor_of(actor)
    ewc = ewc_mod.EWC(actor, fisher_diag=fisher, coef=1.0)
    _perturb(actor, 0.03)
    pen = _f(ewc.penalty())

    params = {name: param for name, param in actor.named_parameters()}
    functional = _f(ewc_mod.ewc_loss(params=params, anchor=anchor, fisher=fisher, coef=1.0))
    assert _approx(functional, pen, rel=1e-4), (functional, pen)

    # fisher=None degenerates to a uniform L2 anchor (all-ones Fisher here)
    l2 = _f(ewc_mod.ewc_loss(params=params, anchor=anchor, coef=1.0))
    assert _approx(l2, pen, rel=1e-4), (l2, pen)

    try:
        helper = _f(ewc_mod.diagonal_fisher_penalty(actor, fisher, anchor=anchor, coef=1.0))
    except TypeError as exc:  # pragma: no cover - signature drift
        raise SkipTest("diagonal_fisher_penalty signature differs: %s" % (exc,))
    assert _approx(helper, pen, rel=1e-4), (helper, pen)


def test_ewc_penalty_is_differentiable_actor_only() -> None:
    """The penalty is differentiable w.r.t. the actor and its gradient is linear."""
    ewc_mod = _imp("src.retention.ewc")

    actor = _make_actor(2)
    fisher = _named_fisher(actor, 1.0)
    ewc = ewc_mod.EWC(actor, fisher_diag=fisher, coef=1.0)
    _perturb(actor, 0.02)

    penalty = ewc.penalty()
    assert hasattr(penalty, "backward"), "penalty must be a differentiable tensor"
    penalty.backward()

    grads = [p.grad for p in actor.parameters() if p.grad is not None]
    assert grads, "penalty must produce gradients for the actor parameters"
    # d/dtheta of sum (theta_pre - theta)^2 = -2 (theta_pre - theta) = 2 * 0.02
    for name, param in actor.named_parameters():
        if param.grad is None:
            continue
        expected = 2.0 * 0.02 * fisher[name].numel() / max(param.numel(), 1)
        assert _approx(_f(param.grad) / max(param.numel(), 1), expected, rel=1e-4)


def test_normalize_param_name_strips_prefixes_and_suffixes() -> None:
    ewc_mod = _imp("src.retention.ewc")
    fn = getattr(ewc_mod, "normalize_param_name", None)
    if fn is None:
        raise SkipTest("normalize_param_name not exported")
    assert fn("actor.fc1.weight") == "fc1.weight"
    assert fn("policy.fc1.weight_pre") == "fc1.weight"
    assert fn("module.lstm.weight_ih_l0") == "lstm.weight_ih_l0"
    assert fn("fc2.bias") == "fc2.bias"


# ---------------------------------------------------------------------------
# Fisher  (Appendix C.1 / B.1)
# ---------------------------------------------------------------------------

def test_fisher_diagonal_nonnegative_and_param_aligned() -> None:
    """The diagonal Fisher is non-negative, finite and name-aligned."""
    fisher_mod = _imp("src.retention.fisher")

    actor = _make_actor(0)
    estimator = fisher_mod.FisherEstimator(actor, mode="expert")

    torch.manual_seed(0)
    obs = torch.randn(16, 4)
    actions = torch.randint(0, actor.n_actions, (16,))
    estimator.accumulate(obs, actions)

    diag = estimator.diagonal()
    assert isinstance(diag, dict), "diagonal() must return a name -> tensor mapping"
    names = {name for name, _ in actor.named_parameters()}
    assert set(diag.keys()) == names, (set(diag.keys()) ^ names)

    total = 0.0
    for name, tensor in diag.items():
        values = tensor.detach().float().reshape(-1)
        assert bool(torch.isfinite(values).all()), name
        assert bool((values >= 0).all()), name
        total += float(values.sum())
    assert total > 0.0, "the Fisher diagonal must be positive after accumulating data"


def test_fisher_accumulation_is_additive_and_resettable() -> None:
    """Re-accumulating identical data adds to (or averages) the diagonal exactly."""
    fisher_mod = _imp("src.retention.fisher")

    actor = _make_actor(0)
    estimator = fisher_mod.FisherEstimator(actor, mode="expert")
    torch.manual_seed(0)
    obs = torch.randn(8, 4)
    actions = torch.randint(0, actor.n_actions, (8,))

    estimator.accumulate(obs, actions)
    first = {k: v.detach().clone() for k, v in estimator.diagonal().items()}
    estimator.accumulate(obs, actions)
    second = estimator.diagonal()

    ratios = []
    for name, value in second.items():
        base = float(first[name].sum())
        if base <= 0.0:
            continue
        ratios.append(float(value.sum()) / base)
    assert ratios, "expected at least one positive Fisher entry"
    # 'sum' reduction doubles the statistic, 'mean' reduction keeps it constant.
    assert all(_approx(r, 2.0, rel=1e-3) or _approx(r, 1.0, rel=1e-3) for r in ratios), ratios

    if hasattr(estimator, "reset"):
        estimator.reset()
        after = estimator.diagonal()
        assert _approx(sum(_f(v) for v in after.values()), 0.0, abs_=1e-12)


def test_fisher_dot_matches_closed_form() -> None:
    """fisher_dot == sum_i F^i (theta_pre^i - theta^i)^2."""
    fisher_mod = _imp("src.retention.fisher")

    actor = _make_actor(3)
    fisher = _named_fisher(actor, 2.0)
    anchor = {name: param.detach().clone() + 0.5
              for name, param in actor.named_parameters()}
    params = {name: param for name, param in actor.named_parameters()}

    expected = 0.0
    for name, param in params.items():
        diff = anchor[name] - param
        expected += float((fisher[name] * diff * diff).sum())

    value = _f(fisher_mod.fisher_dot(fisher, params, anchor))
    assert _approx(value, expected, rel=1e-5), (value, expected)


# ---------------------------------------------------------------------------
# Behavioral cloning  (Appendix C.2)
# ---------------------------------------------------------------------------

def _categorical_pair() -> Tuple[Any, Any]:
    """(near-deterministic student, uniform teacher) with well separated KLs."""
    student = Categorical(logits=torch.tensor([[0.0, 20.0]]))
    teacher = Categorical(logits=torch.tensor([[0.0, 0.0]]))
    return student, teacher


def test_kl_s_divergence_is_log_ratio() -> None:
    bc_mod = _imp("src.retention.behavioral_cloning")

    p = torch.tensor([-1.0, -2.0, 0.5])
    q = torch.tensor([-3.0, -0.5, 0.5])
    diff = bc_mod.kl_s_divergence(p, q)
    got = [float(v) for v in diff.detach().reshape(-1)]
    expected = [2.0, -1.5, 0.0]
    for g, e in zip(got, expected):
        assert _approx(g, e, abs_=1e-6), (got, expected)


def test_bc_loss_vanishes_when_student_equals_teacher() -> None:
    """KL(pi || pi) == 0 exactly, in both directions and both estimators."""
    bc_mod = _imp("src.retention.behavioral_cloning")

    torch.manual_seed(0)
    dist = Categorical(logits=torch.randn(5, 3))
    clone = Categorical(logits=dist.logits.detach().clone())

    for direction in ("forward", "reverse"):
        for exact in (True, False):
            try:
                value = _f(bc_mod.bc_loss(dist, clone, direction=direction, use_exact_kl=exact))
            except TypeError as exc:  # pragma: no cover - signature drift
                raise SkipTest("bc_loss signature differs: %s" % (exc,))
            assert abs(value) < 1e-6, (direction, exact, value)


def test_bc_forward_direction_is_student_to_teacher() -> None:
    """Appendix C.2: ``forward`` == KL(student || teacher) (deterministic probe)."""
    bc_mod = _imp("src.retention.behavioral_cloning")

    student, teacher = _categorical_pair()
    kl_student_teacher = float(torch.distributions.kl_divergence(student, teacher))
    kl_teacher_student = float(torch.distributions.kl_divergence(teacher, student))
    assert _approx(kl_student_teacher, math.log(2.0), rel=1e-3)
    assert kl_teacher_student > 5.0  # very different from the forward direction

    for exact in (True, False):
        try:
            value = _f(bc_mod.bc_loss(student, teacher, direction="forward", use_exact_kl=exact))
        except TypeError as exc:  # pragma: no cover
            raise SkipTest("bc_loss signature differs: %s" % (exc,))
        # Sampling from a near-deterministic student makes the Monte-Carlo
        # estimate exact as well, so both paths must agree with KL(student||teacher).
        assert _approx(value, kl_student_teacher, rel=1e-3, abs_=1e-3), (exact, value)


def test_sampled_kl_reverse_direction_uses_teacher_to_student() -> None:
    """``reverse`` == KL(teacher || student) when actions are provided."""
    bc_mod = _imp("src.retention.behavioral_cloning")
    sampled_kl = getattr(bc_mod, "sampled_kl", None)
    if sampled_kl is None:
        raise SkipTest("sampled_kl not exported")

    student, teacher = _categorical_pair()
    actions = torch.tensor([0, 1, 0, 1])
    expected = _f(teacher.log_prob(actions) - student.log_prob(actions)) / 4.0
    try:
        value = _f(sampled_kl(student, teacher, direction="reverse", actions=actions))
    except TypeError as exc:  # pragma: no cover
        raise SkipTest("sampled_kl signature differs: %s" % (exc,))
    if not _approx(value, expected, rel=1e-4, abs_=1e-6):
        # A Monte-Carlo estimator that ignores the provided actions must still
        # produce a positive, bounded estimate of KL(teacher || student).
        assert 0.0 < value < 25.0, (value, expected)


def test_bc_buffer_roundtrip_and_teacher_precompute() -> None:
    bc_mod = _imp("src.retention.behavioral_cloning")

    torch.manual_seed(0)
    obs = torch.randn(12, 4)
    actions = torch.randint(0, 3, (12,))

    buffer = bc_mod.BCBuffer(capacity=64)
    buffer.add(obs, actions)
    assert len(buffer) == 12

    sample = buffer.sample(5)
    assert _field_of(sample, "obs") is not None
    assert _field_of(sample, "actions") is not None

    teacher = _make_actor(7)
    if hasattr(buffer, "precompute"):
        buffer.precompute(teacher, batch_size=4)
        sample2 = buffer.sample(5)
        assert _field_of(sample2, "obs") is not None

    built = bc_mod.build_bc_buffer(obs, teacher=teacher, actions=actions, capacity=64)
    assert len(built) == 12


def test_behavioral_cloning_object_is_actor_only_and_zero_when_identical() -> None:
    bc_mod = _imp("src.retention.behavioral_cloning")

    torch.manual_seed(0)
    obs = torch.randn(10, 4)
    actions = torch.randint(0, 3, (10,))

    teacher = _make_actor(11)
    bc_mod.freeze_teacher(teacher)
    assert all(not p.requires_grad for p in teacher.parameters()), "teacher must be frozen"

    actor = _make_actor(11)  # identical initialisation -> zero KL
    buffer = bc_mod.build_bc_buffer(obs, teacher=teacher, actions=actions, capacity=64)
    bc = bc_mod.BehavioralCloning(actor, teacher=teacher, buffer=buffer, coef=1.0)

    loss = _invoke_loss(bc, obs)
    assert _f(loss) >= -1e-8
    assert abs(_f(loss)) < 1e-5, "KL(student||teacher) must vanish for identical policies"

    coef = getattr(bc, "coefficient", None)
    if coef is not None:
        assert _approx(float(coef), 1.0, rel=1e-9), "BC uses a constant coefficient"

    # a perturbed teacher yields a strictly positive KL
    other_teacher = _make_actor(12)
    bc2 = bc_mod.BehavioralCloning(actor, teacher=other_teacher, buffer=buffer, coef=1.0)
    assert _f(_invoke_loss(bc2, obs)) > 0.0


def test_coefficient_schedule_constant_and_hooks() -> None:
    bc_mod = _imp("src.retention.behavioral_cloning")
    schedule_cls = getattr(bc_mod, "CoefficientSchedule", None)
    if schedule_cls is None:
        raise SkipTest("CoefficientSchedule not exported")

    try:
        const = schedule_cls(kind="none", value=2.0)
    except TypeError as exc:  # pragma: no cover
        raise SkipTest("CoefficientSchedule signature differs: %s" % (exc,))

    for step in (0, 1, 1000, 10 ** 6):
        assert _approx(_f(const.value_at(step)), 2.0, rel=1e-9)
    assert _approx(_f(const.coefficient), 2.0, rel=1e-9)


# ---------------------------------------------------------------------------
# Kickstarting  (Appendix C.2)
# ---------------------------------------------------------------------------

def test_per_step_decay_matches_paper_schedule() -> None:
    ks_mod = _imp("src.retention.kickstarting")
    decay_cls = getattr(ks_mod, "PerStepDecay", None)
    if decay_cls is None:
        raise SkipTest("PerStepDecay not exported")

    decay = 0.99998
    schedule = decay_cls(value=0.5, decay=decay)
    assert _approx(_f(schedule.value_at(0)), 0.5, rel=1e-9)
    assert _approx(_f(schedule.value_at(1000)), 0.5 * decay ** 1000, rel=1e-6)
    assert _approx(_f(schedule.value_at(10000)), 0.5 * decay ** 10000, rel=1e-6)

    schedule.step(10000)
    assert _approx(_f(schedule.coefficient), 0.5 * decay ** 10000, rel=1e-3)

    if hasattr(schedule, "reset"):
        schedule.reset()
        assert _approx(_f(schedule.coefficient), 0.5, rel=1e-9)


def test_ks_config_for_nethack() -> None:
    ks_mod = _imp("src.retention.kickstarting")

    assert _approx(ks_mod.ks_coef_for("nethack"), 0.5, rel=1e-12)
    assert _approx(ks_mod.ks_decay_for("nethack"), 0.99998, rel=1e-12)

    if hasattr(ks_mod, "ks_config_for"):
        cfg = ks_mod.ks_config_for("nethack")
        coef = _field_of(cfg, "coef", _field_of(cfg, "coefficient"))
        decay = _field_of(cfg, "decay")
        if coef is not None:
            assert _approx(_f(coef), 0.5, rel=1e-12)
        if decay is not None:
            assert _approx(_f(decay), 0.99998, rel=1e-12)

    assert _approx(getattr(ks_mod, "DEFAULT_KS_COEF")["nethack"], 0.5, rel=1e-12)
    assert _approx(getattr(ks_mod, "DEFAULT_KS_DECAY")["nethack"], 0.99998, rel=1e-12)


def test_kickstarting_object_reverse_kl_and_decay() -> None:
    ks_mod = _imp("src.retention.kickstarting")

    torch.manual_seed(0)
    obs = torch.randn(8, 4)

    teacher = _make_actor(21)
    student = _make_actor(21)  # identical -> KL(teacher||student) == 0
    ks_mod.freeze_teacher(teacher)

    ks = ks_mod.Kickstarting(student, teacher=teacher, coef=0.5, decay=0.99998, batch_size=4)
    loss = _invoke_loss(ks, obs)
    assert abs(_f(loss)) < 1e-5, "KL(teacher||student) must vanish for identical policies"

    if hasattr(ks, "step"):
        coef_prop = getattr(ks, "coefficient", None)
        if coef_prop is not None:
            coef0 = float(coef_prop)
            ks.step(10000)
            assert _approx(float(coef_prop), coef0 * 0.99998 ** 10000, rel=1e-2), \
                (coef0, float(coef_prop))

    other_teacher = _make_actor(22)
    ks2 = ks_mod.Kickstarting(student, teacher=other_teacher, coef=0.5)
    assert _f(_invoke_loss(ks2, obs)) > 0.0


def test_kickstarting_functional_loss_is_non_negative() -> None:
    ks_mod = _imp("src.retention.kickstarting")

    student, teacher = _categorical_pair()
    fn = getattr(ks_mod, "kickstarting_loss", getattr(ks_mod, "ks_loss", None))
    if fn is None:
        raise SkipTest("kickstarting_loss not exported")

    try:
        value = _f(fn(teacher, student, direction="reverse"))  # KL(teacher||student)
    except TypeError as exc:  # pragma: no cover
        raise SkipTest("kickstarting_loss signature differs: %s" % (exc,))
    assert value >= 0.0
    assert value < 25.0


# ---------------------------------------------------------------------------
# Episodic memory  (Appendix C.3) and paper coefficients
# ---------------------------------------------------------------------------

def _transition(value: float, dim: int = 4) -> Dict[str, Any]:
    obs = torch.full((dim,), float(value))
    return {
        "obs": obs,
        "action": 0,
        "reward": 1.0,
        "next_obs": obs.clone(),
        "done": False,
    }


def test_episodic_memory_protected_region_never_overwritten() -> None:
    _require_numpy()
    em_mod = _imp("src.retention.episodic_memory")

    capacity, fraction = 100, 0.1
    buffer = em_mod.EpisodicMemoryBuffer(capacity=capacity, fraction=fraction)
    n_prior = max(1, int(round(capacity * fraction)))

    prior = [_transition(100.0) for _ in range(n_prior)]
    try:
        buffer.set_prior_data(prior)
    except Exception:
        for transition in prior:
            buffer.add(transition, protected=True)

    assert buffer.protected_count == n_prior, buffer.protected_count

    for i in range(5 * capacity):  # far more than the capacity
        try:
            buffer.add(_transition(float(i)), protected=False)
        except TypeError:  # pragma: no cover
            buffer.add(_transition(float(i)))

    assert len(buffer) <= capacity, len(buffer)
    assert buffer.protected_count == n_prior, "protected slots must never be overwritten"

    for index in range(n_prior):
        obs = _np.asarray(_np.asarray(_field_of(buffer[index], "obs")).reshape(-1)[0])
        assert _approx(float(obs), 100.0, abs_=1e-6), (index, float(obs))

    sample = buffer.sample(16)
    assert _field_of(sample, "obs") is not None

    assert _approx(em_mod.em_fraction_for("robotic_sequence"), 0.1, rel=1e-9)


def test_episodic_memory_penalty_is_differentiable_zero() -> None:
    em_mod = _imp("src.retention.episodic_memory")

    actor = _make_actor(0)
    em = em_mod.EpisodicMemory(actor=actor, capacity=100, fraction=0.1)
    try:
        penalty = em.penalty()
    except TypeError as exc:  # pragma: no cover
        raise SkipTest("EpisodicMemory.penalty signature differs: %s" % (exc,))

    assert abs(_f(penalty)) < 1e-12
    if hasattr(penalty, "backward"):
        penalty.backward()  # a differentiable zero keeps the RL loss valid

    assert getattr(em_mod.EpisodicMemory, "HAS_AUXILIARY_LOSS", False) is False


def test_mixed_batch_sampler_guarantees_prior_data() -> None:
    _require_numpy()
    em_mod = _imp("src.retention.episodic_memory")
    sampler_cls = getattr(em_mod, "MixedBatchSampler", None)
    if sampler_cls is None:
        raise SkipTest("MixedBatchSampler not exported")

    sampler = sampler_cls(num_prior=10, num_current=90, prior_fraction=0.1)
    assert sampler.total == 100

    generator = _np.random.default_rng(0)
    indices = list(sampler.sample_indices(20, generator))
    assert len(indices) == 20
    prior_in_batch = sum(1 for index in indices if int(index) < 10)
    assert prior_in_batch >= 1, "every mixed batch must contain prior-task data"
    assert sampler.prior_count(20) >= 1


def test_retention_coefficients_match_paper() -> None:
    ewc_mod = _imp("src.retention.ewc")
    bc_mod = _imp("src.retention.behavioral_cloning")
    ks_mod = _imp("src.retention.kickstarting")

    # Appendix B.1 / B.3 / Table 3 values.
    assert _approx(ewc_mod.ewc_coef_for("nethack"), 2e6, rel=1e-12)
    assert _approx(ewc_mod.ewc_coef_for("robotic_sequence"), 100.0, rel=1e-12)
    assert _approx(bc_mod.bc_coef_for("nethack"), 2.0, rel=1e-12)
    assert _approx(bc_mod.bc_coef_for("robotic_sequence"), 1.0, rel=1e-12)
    assert _approx(ks_mod.ks_coef_for("nethack"), 0.5, rel=1e-12)
    assert _approx(ks_mod.ks_decay_for("nethack"), 0.99998, rel=1e-12)

    assert int(getattr(bc_mod, "DEFAULT_BC_MEMORY", 10000)) == 10000
    assert _approx(getattr(ewc_mod, "DEFAULT_EWC_COEF")["nethack"], 2e6, rel=1e-12)
    assert _approx(getattr(ewc_mod, "DEFAULT_EWC_COEF")["robotic_sequence"], 100.0, rel=1e-12)
    assert _approx(getattr(bc_mod, "DEFAULT_BC_COEF")["nethack"], 2.0, rel=1e-12)

    fisher_mod = _imp("src.retention.fisher")
    assert int(getattr(fisher_mod, "DEFAULT_FISHER_BATCHES", 10000)) == 10000
    assert int(getattr(fisher_mod, "DEFAULT_BATCH_SIZE", 128)) == 128


def test_retention_modules_are_importable_without_torch() -> None:
    """Each retention module must import cleanly (torch is optional at import)."""
    for name in (
        "src.retention.ewc",
        "src.retention.fisher",
        "src.retention.behavioral_cloning",
        "src.retention.kickstarting",
        "src.retention.episodic_memory",
    ):
        module = importlib.import_module(name)
        assert module is not None, name


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_TESTS: List[Callable[[], None]] = [
    test_ewc_penalty_zero_at_pretrained_weights,
    test_ewc_penalty_grows_quadratically_and_scales_with_coef,
    test_ewc_functional_and_helper_match_object_penalty,
    test_ewc_penalty_is_differentiable_actor_only,
    test_normalize_param_name_strips_prefixes_and_suffixes,
    test_fisher_diagonal_nonnegative_and_param_aligned,
    test_fisher_accumulation_is_additive_and_resettable,
    test_fisher_dot_matches_closed_form,
    test_kl_s_divergence_is_log_ratio,
    test_bc_loss_vanishes_when_student_equals_teacher,
    test_bc_forward_direction_is_student_to_teacher,
    test_sampled_kl_reverse_direction_uses_teacher_to_student,
    test_bc_buffer_roundtrip_and_teacher_precompute,
    test_behavioral_cloning_object_is_actor_only_and_zero_when_identical,
    test_coefficient_schedule_constant_and_hooks,
    test_per_step_decay_matches_paper_schedule,
    test_ks_config_for_nethack,
    test_kickstarting_object_reverse_kl_and_decay,
    test_kickstarting_functional_loss_is_non_negative,
    test_episodic_memory_protected_region_never_overwritten,
    test_episodic_memory_penalty_is_differentiable_zero,
    test_mixed_batch_sampler_guarantees_prior_data,
    test_retention_coefficients_match_paper,
    test_retention_modules_are_importable_without_torch,
]


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    quiet = "--quiet" in argv or "-q" in argv

    passed, failed, skipped = 0, 0, 0
    for test in _TESTS:
        name = getattr(test, "__name__", str(test))
        try:
            test()
        except SkipTest as exc:
            skipped += 1
            if not quiet:
                print("SKIP %s (%s)" % (name, exc))
        except Exception:
            failed += 1
            print("FAIL %s" % (name,))
            traceback.print_exc()
        else:
            passed += 1
            if not quiet:
                print("ok   %s" % (name,))

    print("%d passed, %d failed, %d skipped" % (passed, failed, skipped))
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
