"""Unit tests of the FOA building blocks (Eqn. (5)-(9)) - fast, CPU only."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from foa.config import FOAConfig
from foa.core.activation_shift import BackToSourceShifting
from foa.core.cma import NumpyCMA, make_cma
from foa.core.fitness import (
    activation_discrepancy,
    foa_fitness,
    prediction_entropy,
)
from foa.core.foa import FOA, FOAInterval
from foa.core.statistics import FeatureStatistics, compute_source_statistics
from foa.evaluation.metrics import accuracy, expected_calibration_error


# --------------------------------------------------------------------------------------
# CMA-ES
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("backend", ["numpy", "cmaes"])
def test_cma_minimises_sphere(backend):
    dim, popsize = 20, 16
    if backend == "numpy":
        opt = NumpyCMA(np.zeros(dim), sigma0=1.0, popsize=popsize, seed=0)
    else:
        opt = make_cma(np.zeros(dim), sigma0=1.0, popsize=popsize, seed=0)
    assert opt.popsize == popsize
    best = np.inf
    for _ in range(250):
        solutions = opt.ask()
        assert solutions.shape == (popsize, dim)
        values = (solutions**2).sum(axis=1)
        best = min(best, float(values.min()))
        opt.tell(solutions, values)
    assert best < 1e-3
    assert np.linalg.norm(opt.mean) < 1e-2


def test_cma_updates_the_distribution():
    opt = NumpyCMA(np.zeros(10), sigma0=1.0, popsize=12, seed=1)
    mean_before = opt.mean.copy()
    sigma_before = opt.sigma
    solutions = opt.ask()
    opt.tell(solutions, (solutions - 1.0) ** 2 @ np.ones(10))
    assert not np.allclose(mean_before, opt.mean)
    assert opt.sigma != sigma_before


# --------------------------------------------------------------------------------------
# fitness function (Eqn. 5)
# --------------------------------------------------------------------------------------
def test_prediction_entropy_matches_definition():
    logits = torch.randn(8, 5)
    manual = -(F.softmax(logits, -1) * F.log_softmax(logits, -1)).sum(-1)
    assert torch.allclose(prediction_entropy(logits, reduction="none"), manual, atol=1e-6)
    assert torch.allclose(prediction_entropy(logits, reduction="sum"), manual.sum(), atol=1e-5)


def test_activation_discrepancy_and_fitness():
    torch.manual_seed(0)
    feats = [torch.randn(6, 4) for _ in range(4)]        # e_0 .. e_3
    stats = FeatureStatistics(
        [torch.zeros(4) for _ in range(4)], [torch.ones(4) for _ in range(4)]
    )
    expected = torch.zeros(())
    for i in range(1, 4):
        mu = feats[i].mean(0)
        sd = feats[i].std(0, unbiased=False)
        expected += (mu - 0).norm() + (sd - 1).norm()
    got = activation_discrepancy(feats, stats, layers=[1, 2, 3])
    assert torch.allclose(got, expected, atol=1e-5)

    logits = torch.randn(6, 5)
    lam = 0.4
    value = foa_fitness(logits, feats, stats, lam, layers=[1, 2, 3])
    assert torch.allclose(value, prediction_entropy(logits) + lam * expected, atol=1e-5)
    # entropy-only and discrepancy-only special cases
    assert torch.allclose(
        foa_fitness(logits, feats, stats, lam, use_activation_discrepancy=False),
        prediction_entropy(logits),
        atol=1e-6,
    )


# --------------------------------------------------------------------------------------
# activation shifting (Eqn. 7-9)
# --------------------------------------------------------------------------------------
def test_activation_shifting_ema():
    source_mean = torch.zeros(4)
    shift = BackToSourceShifting(source_mean, alpha=0.1, gamma=1.0)
    first = torch.ones(4)
    d1 = shift.update(first)                       # mu_N(0) = mu_N(X_1) (addendum)
    assert torch.allclose(d1, -first)
    second = 2 * first
    shift.update(second)
    ema = 0.1 * second + 0.9 * first               # Eqn. (9)
    assert torch.allclose(shift.ema, ema)
    assert torch.allclose(shift.direction, source_mean - ema)
    # gamma scales the shift
    shift2 = BackToSourceShifting(source_mean, alpha=0.1, gamma=0.5)
    assert torch.allclose(shift2.update(first), -0.5 * first)


def test_shift_is_applied_to_the_final_activation(tiny_model, tiny_images):
    prompt = torch.zeros(3, tiny_model.embed_dim)
    _, e_ref, _ = tiny_model.forward_with_prompt(tiny_images, prompt=prompt)
    shift = torch.arange(tiny_model.embed_dim, dtype=torch.float32) * 1e-3
    _, e_shift, feats = tiny_model.forward_with_prompt(
        tiny_images, prompt=prompt, shift=shift, return_layers=True
    )
    assert torch.allclose(e_shift - e_ref, shift.expand_as(e_shift), atol=1e-6)
    # the last layer feature is the shifted activation (Alg. 1: adjust -> predict)
    assert torch.allclose(feats[-1], e_shift)


# --------------------------------------------------------------------------------------
# prompts / layer features
# --------------------------------------------------------------------------------------
def test_prompt_shapes_and_layer_features(tiny_model, tiny_images):
    B = tiny_images.shape[0]
    logits, e_n, feats = tiny_model.forward_with_prompt(
        tiny_images, prompt=torch.randn(3, tiny_model.embed_dim), return_layers=True
    )
    assert logits.shape == (B, tiny_model.num_classes)
    assert e_n.shape == (B, tiny_model.embed_dim)
    # one feature per layer plus the input embedding (i = 0 .. N)
    assert len(feats) == tiny_model.num_layers + 1
    assert all(f.shape == (B, tiny_model.embed_dim) for f in feats)
    assert tiny_model.prompt_dim == 3 * tiny_model.embed_dim


def test_preprocess_and_forward_tokens_are_equivalent(tiny_model, tiny_images):
    tokens = tiny_model.preprocess(tiny_images)
    prompt = torch.randn(3, tiny_model.embed_dim)
    a, _, _ = tiny_model.forward_with_prompt(tiny_images, prompt=prompt)
    b, _, _ = tiny_model.forward_tokens(tokens, prompt=prompt)
    assert torch.allclose(a, b, atol=1e-6)


# --------------------------------------------------------------------------------------
# source statistics
# --------------------------------------------------------------------------------------
def test_source_statistics(tiny_model, tiny_images):
    stats = compute_source_statistics(tiny_model, [tiny_images, tiny_images])
    assert stats.num_layers == tiny_model.num_layers + 1
    assert stats.means[0].shape == (tiny_model.embed_dim,)
    assert torch.all(stats.stds[0] >= 0)


# --------------------------------------------------------------------------------------
# FOA end to end
# --------------------------------------------------------------------------------------
def test_foa_step_updates_cma_and_returns_predictions(tiny_model, tiny_images):
    stats = compute_source_statistics(tiny_model, [tiny_images])
    foa = FOA(tiny_model, stats, FOAConfig(popsize=4, batch_size=4), device=torch.device("cpu"))
    mean_before = foa.es.mean.copy()
    out = foa.step(tiny_images)
    assert out.logits.shape == (4, tiny_model.num_classes)
    assert out.fitness.shape == (4,)
    assert out.prompt.numel() == tiny_model.prompt_dim
    assert out.shift is not None and out.shift.shape == (tiny_model.embed_dim,)
    assert not np.allclose(mean_before, foa.es.mean)
    assert foa.iteration == 1


def test_foa_interval_buffers_samples(tiny_model, tiny_images):
    stats = compute_source_statistics(tiny_model, [tiny_images])
    cfg = FOAConfig(popsize=3, interval=2, interval_store="feature")
    foa = FOAInterval(tiny_model, stats, cfg=cfg, device=torch.device("cpu"))
    assert foa.step(tiny_images[:1]) is None
    out = foa.step(tiny_images[1:2])
    assert out is not None and out.logits.shape[0] == 2
    assert foa.pending == 0
    # a partial interval is flushed at the end of the stream
    assert foa.step(tiny_images[:1]) is None
    out = foa.flush()
    assert out is not None and out.logits.shape[0] == 1


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------
def test_accuracy_and_ece():
    logits = torch.tensor([[10.0, 0.0], [0.0, 10.0], [10.0, 0.0], [0.0, 10.0]])
    targets = torch.tensor([0, 0, 0, 1])
    assert accuracy(logits, targets) == pytest.approx(75.0)
    # one confidence bin holds all four samples: |acc - conf| = |0.75 - 1.0| = 25%
    assert expected_calibration_error(logits, targets) == pytest.approx(25.0, abs=0.1)
    # a uniform predictor has confidence 0.5; with 50% accuracy the ECE is 0
    uniform = torch.zeros(4, 2)
    assert expected_calibration_error(uniform, torch.tensor([0, 1, 0, 1])) == pytest.approx(0.0, abs=1e-3)
    assert expected_calibration_error(uniform, targets) == pytest.approx(25.0, abs=0.1)
