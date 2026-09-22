"""Offline unit tests for the BBox-Adapter ranking-based NCE objective.

These tests validate the mathematical contracts of
``bbox_adapter.losses.nce`` (Eq. 1 softmax posterior, Eq. 2 ranking NCE loss,
Eq. 3 four-term gradient/objective) and of ``bbox_adapter.adapter.regularizer``
(``alpha * E[g^2]`` energy penalty, notably *not* power iteration).

They require only ``torch`` and the local package; no network, no black-box LLM,
no ``transformers`` download.  Run standalone::

    python tests/test_nce_loss.py

or via pytest::

    pytest tests/test_nce_loss.py -v
"""

from __future__ import annotations

import math
import os
import sys
import traceback
from typing import Any, Dict, List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# --------------------------------------------------------------------------- #
# torch guard
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - environment dependent
    import torch

    _TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _TORCH = False


try:
    from bbox_adapter.losses import nce as N
    from bbox_adapter.adapter import regularizer as R
except Exception as exc:  # pragma: no cover
    print("FATAL: could not import bbox_adapter losses/regularizer:", exc)
    raise


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _t(values, dtype=None):
    """Build a 1-D float tensor from a python sequence."""
    return torch.tensor(list(values), dtype=dtype or torch.float32)


def _close(a: float, b: float, tol: float = 1e-5) -> bool:
    return abs(float(a) - float(b)) <= tol


def _all_close(a, b, tol: float = 1e-5) -> bool:
    return bool(torch.allclose(torch.as_tensor(a, dtype=torch.float32),
                               torch.as_tensor(b, dtype=torch.float32),
                               atol=tol, rtol=0.0))


def _skip(name: str) -> None:
    print("  SKIP %s (torch unavailable)" % name)


# --------------------------------------------------------------------------- #
# Eq. (1): softmax posterior
# --------------------------------------------------------------------------- #
def test_softmax_posterior_normalizes():
    energies = _t([0.0, 1.0, 2.0])
    p = N.softmax_posterior(energies)
    assert _close(float(p.sum()), 1.0), float(p.sum())
    # energies are shifted so the argmax keeps the largest mass
    assert int(torch.argmax(p).item()) == 2
    # closed form check for [0,1,2]
    expected = [math.exp(0.0), math.exp(1.0), math.exp(2.0)]
    total = sum(expected)
    assert _all_close(p, [e / total for e in expected], 1e-6), p


def test_log_softmax_posterior_matches_log():
    energies = _t([0.3, -1.2, 4.0, 0.0])
    lp = N.log_softmax_posterior(energies)
    p = N.softmax_posterior(energies)
    assert _all_close(lp, torch.log(p), 1e-5), (lp, p)
    assert _all_close(lp.exp().sum(), torch.tensor(1.0), 1e-5)


def test_softmax_posterior_is_shift_invariant():
    energies = _t([1.0, 2.0, 3.0])
    shifted = energies + 100.0
    assert _all_close(N.softmax_posterior(energies),
                      N.softmax_posterior(shifted), 1e-5)


def test_softmax_posterior_mask_excludes_entries():
    energies = _t([1.0, 5.0, 2.0])
    mask = torch.tensor([True, False, True])
    p = N.softmax_posterior(energies, mask=mask)
    assert _close(float(p[1]), 0.0), p
    assert _close(float(p.sum()), 1.0), p


# --------------------------------------------------------------------------- #
# Eq. (2): ranking NCE loss = -log softmax(positive) over the contrastive set
# --------------------------------------------------------------------------- #
def test_compute_nce_loss_matches_manual_softmax():
    """Eq. (2) with one positive + K negatives per query."""
    pos = _t([1.0, 0.5])
    negs = _t([[0.0, -1.0], [0.2, 0.1]])
    loss = N.compute_nce_loss(pos, negs, reduction="mean")

    manual = []
    for i in range(2):
        row = [float(pos[i])] + [float(v) for v in negs[i]]
        m = max(row)
        logZ = m + math.log(sum(math.exp(v - m) for v in row))
        manual.append(logZ - float(pos[i]))
    expected = sum(manual) / len(manual)
    assert _close(float(loss), expected, 1e-5), (float(loss), expected)


def test_compute_nce_loss_positive_dominates_gives_zero():
    """If the positive energy is huge, its posterior -> 1 so the loss -> 0."""
    pos = _t([50.0])
    negs = _t([[0.0, -3.0, 1.0]])
    loss = N.compute_nce_loss(pos, negs)
    assert float(loss) < 1e-6, float(loss)


def test_compute_nce_loss_positive_dominated_gives_large_loss():
    pos = _t([-50.0])
    negs = _t([[10.0, 5.0]])
    loss = N.compute_nce_loss(pos, negs)
    assert float(loss) > 20.0, float(loss)


def test_compute_nce_loss_is_lower_with_better_ranking():
    negs = _t([[0.0, 0.0, 0.0]])
    good = N.compute_nce_loss(_t([3.0]), negs)
    bad = N.compute_nce_loss(_t([-3.0]), negs)
    assert float(good) < float(bad), (float(good), float(bad))


def test_compute_nce_loss_reduction_sum_vs_mean():
    pos = _t([1.0, 1.0])
    negs = _t([[0.0, 0.0], [0.0, 0.0]])
    mean = N.compute_nce_loss(pos, negs, reduction="mean")
    total = N.compute_nce_loss(pos, negs, reduction="sum")
    assert _close(float(total), 2.0 * float(mean), 1e-5), (float(total), float(mean))


def test_compute_nce_loss_temperature_scaling():
    """temperature T divides the energies: loss(T) == loss(1) on g/T."""
    pos = _t([2.0])
    negs = _t([[0.0, 1.0]])
    l1 = N.compute_nce_loss(pos, negs, temperature=1.0)
    l2 = N.compute_nce_loss(pos / 2.0, negs / 2.0, temperature=1.0)
    lT = N.compute_nce_loss(pos, negs, temperature=2.0)
    assert _close(float(l2), float(lT), 1e-5), (float(l2), float(lT))
    assert float(l1) > float(lT)


def test_compute_nce_loss_accepts_ragged_negative_lists():
    pos = _t([1.0, 1.0])
    negs = [[0.0, -1.0, 0.5], [0.0]]
    loss = N.compute_nce_loss(pos, negs)
    assert torch.isfinite(loss), loss
    assert float(loss) > 0.0


def test_compute_nce_loss_empty_negatives_reduces_to_zero():
    """With no negatives the posterior is degenerate -> loss is 0 (not NaN)."""
    loss = N.compute_nce_loss(_t([1.0]), None)
    assert torch.isfinite(loss), loss
    assert _close(float(loss), 0.0, 1e-6), float(loss)


def test_label_smoothing_does_not_increase_loss_when_confident():
    pos = _t([20.0])
    negs = _t([[0.0]])
    plain = N.compute_nce_loss(pos, negs, label_smoothing=0.0)
    smooth = N.compute_nce_loss(pos, negs, label_smoothing=0.1)
    assert float(smooth) >= float(plain) - 1e-6, (float(smooth), float(plain))


def test_ranking_nce_loss_alias_and_module_wrapper():
    pos = _t([1.0])
    negs = _t([[0.0, -1.0]])
    a = N.compute_nce_loss(pos, negs)
    b = N.ranking_nce_loss(pos, negs)
    assert _close(float(a), float(b), 1e-6)


def test_ranking_nceloss_module_forward():
    loss_fn = N.RankingNCELoss(alpha=0.0, use_regularizer=False)
    pos = _t([1.0, 2.0])
    negs = _t([[0.0, 0.0], [0.0, 0.0]])
    out = loss_fn(positive_energies=pos, negative_energies=negs)
    ref = N.compute_nce_loss(pos, negs)
    assert _close(float(out), float(ref), 1e-5), (float(out), float(ref))


def test_ranking_nceloss_config_roundtrip():
    cfg = N.NCELossConfig(alpha=0.05, reduction="mean", temperature=1.0,
                          use_regularizer=True, reg_mode="split")
    payload = cfg.to_dict()
    back = N.NCELossConfig.from_dict(payload)
    assert _close(back.alpha, 0.05), back.alpha
    assert back.reg_mode == "split", back.reg_mode


# --------------------------------------------------------------------------- #
# Eq. (3): gradient structure + regularizer
# --------------------------------------------------------------------------- #
def test_nce_gradient_sign_structure():
    """Eq. (3): positive energies get negative gradient, negatives positive."""
    pos = torch.tensor([1.0], requires_grad=True)
    neg = torch.tensor([[0.0, 0.0]], requires_grad=True)
    loss = N.compute_nce_loss(pos, neg)
    loss.backward()
    assert float(pos.grad) < 0.0, float(pos.grad)
    assert bool((neg.grad > 0.0).all()), neg.grad
    # softmax posterior of the positive equals |dL/dg+|
    p = N.softmax_posterior(torch.cat([pos.detach(), neg.detach().view(-1)]))
    assert _close(float(pos.grad), float(p[0] - 1.0), 1e-5), (float(pos.grad), float(p[0]))


def test_nce_gradient_terms_keys_and_values():
    pos = _t([1.0])
    neg = _t([[0.0, 0.0]])
    terms = N.nce_gradient_terms(pos, neg, alpha=0.0)
    for key in ("neg_data", "reg_pos", "neg_model", "reg_neg", "total"):
        assert key in terms, (key, sorted(terms))
    # Eq. (3): -E[g+] + alpha E[g+^2] + E_{p_theta}[g-] + alpha E[g-^2]
    assert _close(float(terms["neg_data"]), -1.0, 1e-5), terms["neg_data"]
    assert _close(float(terms["reg_pos"]), 0.0, 1e-6), terms["reg_pos"]
    assert _close(float(terms["total"]),
                  float(terms["neg_data"]) + float(terms["reg_pos"])
                  + float(terms["neg_model"]) + float(terms["reg_neg"]), 1e-5)


def test_nce_gradient_terms_regularization_is_positive():
    pos = _t([2.0])
    neg = _t([[1.0]])
    terms = N.nce_gradient_terms(pos, neg, alpha=1e-2)
    assert float(terms["reg_pos"]) > 0.0, terms["reg_pos"]
    assert float(terms["reg_neg"]) > 0.0, terms["reg_neg"]
    assert _close(float(terms["reg_pos"]), 1e-2 * 4.0, 1e-6)


def test_regularizer_alpha_zero_is_noop():
    pos = _t([3.0])
    neg = _t([[-2.0]])
    base = N.compute_nce_loss(pos, neg)
    reg = R.EnergyRegularizer(alpha=0.0)
    combined, stats = reg.combine(base, pos, neg)
    assert _close(float(combined), float(base), 1e-6), (float(combined), float(base))


def test_regularizer_squared_energy_value():
    pos = _t([2.0, 4.0])
    neg = _t([[1.0, 0.0]])
    pen = R.squared_energy_penalty(pos, alpha=0.5, reduction="mean")
    assert _close(float(pen), 0.5 * ((4.0 + 16.0) / 2.0), 1e-5), float(pen)
    pn = R.positive_negative_penalty(pos, neg, alpha=0.5)
    assert _close(float(pn[0]), 0.5 * 10.0, 1e-5), pn
    assert _close(float(pn[1]), 0.5 * 0.5, 1e-5), pn


def test_regularizer_split_equals_sum_of_group_terms():
    pos = _t([1.0, 2.0])
    neg = _t([[0.5, -0.5], [3.0, 0.0]])
    reg = R.EnergyRegularizer(alpha=1e-2, mode="split")
    total = reg.penalty(positive_energies=pos, negative_energies=neg)
    p, n = reg.split_terms(pos, neg)
    assert _all_close(total, p + n, 1e-6), (total, p, n)


def test_regularizer_increases_combined_loss():
    pos = _t([1.0])
    neg = _t([[0.0]])
    base = N.compute_nce_loss(pos, neg)
    reg = R.EnergyRegularizer(alpha=0.1)
    combined, stats = reg.combine(base, pos, neg)
    assert float(combined) > float(base), (float(combined), float(base))
    for key in ("reg", "reg_pos", "reg_neg", "alpha"):
        assert key in stats, (key, sorted(stats))


def test_nce_objective_with_regularizer_matches_manual():
    pos = _t([1.0])
    neg = _t([[0.0]])
    alpha = 0.05
    obj = N.nce_objective(pos, neg, alpha=alpha)
    manual = (float(N.compute_nce_loss(pos, neg))
              + alpha * float(pos.pow(2).mean())
              + alpha * float(neg.pow(2).mean()))
    assert _close(float(obj), manual, 1e-5), (float(obj), manual)


def test_nce_objective_zero_alpha_matches_plain_nce():
    pos = _t([0.7])
    neg = _t([[0.1, -0.2]])
    a = N.nce_objective(pos, neg, alpha=0.0)
    b = N.compute_nce_loss(pos, neg)
    assert _close(float(a), float(b), 1e-6)


def test_regularizer_is_differentiable_and_gradient_positive_on_active():
    """d/dg alpha*g^2 = 2*alpha*g, positive for g > 0."""
    pos = torch.tensor([2.0], requires_grad=True)
    pen = R.squared_energy_penalty(pos, alpha=1.0)
    pen.backward()
    assert _close(float(pos.grad), 4.0, 1e-5), float(pos.grad)


def test_regularizer_alpha_sweep_values():
    assert tuple(R.ALPHA_SWEEP) == (1e-3, 1e-2, 1e-1), R.ALPHA_SWEEP
    assert _close(R.DEFAULT_ALPHA, 1e-2), R.DEFAULT_ALPHA
    assert _close(R.alpha_schedule(1e-2, step=0, warmup_steps=100), 0.0, 1e-12)
    assert _close(R.alpha_schedule(1e-2, step=100, warmup_steps=100), 1e-2, 1e-9)
    assert _close(R.alpha_schedule(1e-2, step=500, warmup_steps=100), 1e-2, 1e-9)


def test_regularizer_check_energy_scales_flags_nonfinite():
    pos = _t([1.0, 2.0])
    neg = _t([[-1.0]])
    stats = R.check_energy_scales(pos, neg, alpha=1e-2)
    assert stats.get("finite", 1.0) == 1.0, stats
    bad = R.check_energy_scales(_t([float("inf")]), None, alpha=1e-2)
    assert bad.get("finite", 1.0) == 0.0, bad


# --------------------------------------------------------------------------- #
# contrastive-set plumbing / diagnostics
# --------------------------------------------------------------------------- #
def test_build_contrastive_tensors_pads_ragged_negatives():
    pos = _t([1.0, 2.0])
    negs = [[0.0, -1.0, 0.5], [0.0]]
    p, n, mask = N.build_contrastive_tensors(pos, negs)
    assert tuple(n.shape) == (2, 3), n.shape
    assert tuple(p.shape) == (2,), p.shape
    assert bool(mask[1].tolist()) is True
    assert mask[1].sum().item() == 1, mask
    assert mask[0].sum().item() == 3, mask


def test_is_list_negatives_detector():
    assert N.is_list_negatives([[0.0, 1.0], [2.0]]) is True
    assert N.is_list_negatives(_t([[0.0, 1.0], [2.0]])) is False


def test_ranking_accuracy_counts_queries():
    pos = _t([2.0, -1.0])
    negs = _t([[1.0, 0.0], [0.0, 0.5]])
    acc = N.ranking_accuracy(pos, negs)
    assert _close(acc, 0.5, 1e-9), acc


def test_ranking_accuracy_all_correct():
    pos = _t([5.0])
    negs = _t([[0.0, 1.0]])
    assert _close(N.ranking_accuracy(pos, negs), 1.0, 1e-9)


def test_binary_nce_and_pairwise_ranking_are_finite():
    pos = _t([1.0, 1.0])
    neg = _t([[0.0], [2.0]])
    b = N.binary_nce_loss(pos, neg)
    m = N.pairwise_ranking_loss(pos, neg, margin=0.0)
    assert torch.isfinite(b) and torch.isfinite(m), (b, m)
    assert float(m) >= 0.0, float(m)


def test_power_iteration_is_not_used():
    """Addendum: the paper's 'spectral normalization' is an L2 energy penalty."""
    src = open(os.path.join(_ROOT, "bbox_adapter", "adapter", "regularizer.py"),
               "r", encoding="utf-8").read().lower()
    assert "power_iteration" not in src, "regularizer must not implement power iteration"
    assert "spectral_norm" not in src, "regularizer must not use torch spectral_norm"


# --------------------------------------------------------------------------- #
# standalone runner
# --------------------------------------------------------------------------- #
def run_all(verbose: bool = True) -> Dict[str, Any]:
    passed: List[str] = []
    failed: List[str] = []
    errors: List[str] = []

    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]

    for name, fn in tests:
        if not _TORCH:
            _skip(name)
            continue
        try:
            fn()
            passed.append(name)
            if verbose:
                print("  PASS %s" % name)
        except AssertionError as exc:
            failed.append(name)
            print("  FAIL %s: %s" % (name, exc))
        except Exception:  # pragma: no cover
            errors.append(name)
            print("  ERROR %s:\n%s" % (name, traceback.format_exc()))

    print("\n%d passed, %d failed, %d errors" % (len(passed), len(failed), len(errors)))
    return {"passed": passed, "failed": failed, "errors": errors}


if __name__ == "__main__":
    if not _TORCH:
        print("torch not available - NCE loss tests cannot run")
        sys.exit(0)
    result = run_all()
    sys.exit(1 if (result["failed"] or result["errors"]) else 0)
