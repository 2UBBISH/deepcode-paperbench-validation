import numpy as np
import torch

from wwmf.config import ExperimentConfig
from wwmf.forecasting.base import ForecastContext, frequency_prior, sample_pairs
from wwmf.forecasting.logit_change import (
    margin_loss,
    pad_to,
    predict_forgetting_from_logits,
)
from wwmf.forecasting.threshold import ThresholdForecaster
from wwmf.forecasting.types import (
    CandidateVocab,
    OnlineArtifact,
    UpstreamCache,
    UpstreamExampleCache,
)
from wwmf.data.types import Example


def _make_cache(n: int = 6, T: int = 3, C: int = 5, all_correct: bool = True) -> UpstreamCache:
    items = []
    for j in range(n):
        items.append(
            UpstreamExampleCache(
                example=Example(f"in{j}", f"out{j}", task="t", idx=j),
                reduced_logits=np.zeros((T, C), dtype=np.float16),
                gold_ids=np.zeros(T, dtype=np.int64),
                mask=np.ones(T, dtype=bool),
                decoder_reps=np.zeros((T, 4), dtype=np.float16),
                pooled_rep=np.zeros(4, dtype=np.float16),
                correct=all_correct if j % 2 == 0 else False,
            )
        )
    return UpstreamCache(
        examples=[it.example for it in items],
        vocab=CandidateVocab(np.arange(C, dtype=np.int64), topk=2, max_size=C),
        items=items,
    )


def _ctx(cache: UpstreamCache, labels: np.ndarray) -> ForecastContext:
    cfg = ExperimentConfig()
    artifacts = [
        OnlineArtifact(
            example=Example(f"x{i}", f"y{i}", task="r", idx=i),
            delta_logits=np.zeros((3, cache.vocab.__len__()), dtype=np.float16),
            gold_ids=np.zeros(3, dtype=np.int64),
            mask=np.ones(3, dtype=bool),
            labels=labels[i],
        )
        for i in range(labels.shape[0])
    ]
    return ForecastContext(cfg=cfg, upstream_cache=cache, train_artifacts=artifacts, verbose=False)


def test_threshold_tuning_maximises_train_f1():
    # the first upstream example is forgotten by all online examples, the last by
    # none: gamma = 1 separates them perfectly.
    labels = np.zeros((5, 4), dtype=np.int8)
    labels[:, 0] = 1
    labels[:, 1] = 1
    labels[3:, 1] = 0
    ctx = _ctx(_make_cache(n=4), labels)
    forecaster = ThresholdForecaster().fit(ctx)
    assert forecaster.gamma >= 1
    pred = forecaster.predict(ctx.train_artifacts[0])
    assert pred[0] == 1
    assert pred[3] == 0


def test_threshold_predictions_do_not_depend_on_online_example():
    labels = np.zeros((4, 3), dtype=np.int8)
    labels[0, 0] = 1
    ctx = _ctx(_make_cache(n=3), labels)
    forecaster = ThresholdForecaster().fit(ctx)
    assert np.array_equal(
        forecaster.predict(ctx.train_artifacts[0]), forecaster.predict(ctx.train_artifacts[2])
    )


def test_threshold_gamma_matches_brute_force_f1_maximisation():
    from wwmf.evaluation.metrics import precision_recall_f1

    rng = np.random.default_rng(0)
    labels = (rng.random((9, 7)) < 0.2).astype(np.int8)
    ctx = _ctx(_make_cache(n=7), labels)
    fitted = ThresholdForecaster().fit(ctx)
    counts = labels.sum(axis=0)
    best_gamma, best_f1 = None, -1.0
    for gamma in range(1, labels.shape[0] + 1):
        pred = np.broadcast_to(counts >= gamma, labels.shape).astype(int)
        _, _, f1 = precision_recall_f1(labels.reshape(-1), pred.reshape(-1))
        if f1 > best_f1:
            best_gamma, best_f1 = gamma, f1
    _, _, fitted_f1 = precision_recall_f1(
        labels.reshape(-1), np.broadcast_to(fitted.predict(ctx.train_artifacts[0]), labels.shape).reshape(-1)
    )
    assert abs(fitted_f1 - best_f1) < 1e-9


def test_pad_to_pads_and_truncates():
    a = np.ones((2, 3))
    assert pad_to(a, 4).shape == (4, 3)
    assert pad_to(a, 1).shape == (1, 3)
    assert pad_to(a, 4)[3].sum() == 0


def test_predict_forgetting_from_logits():
    # vocab ids 10..12, gold token = 10; one example where the arg-max is 11
    vocab = np.array([10, 11, 12])
    gold = np.array([[10, 10]])
    mask = np.array([[True, True]])
    pred_flip = np.array([[[0.0, 5.0, 1.0], [9.0, 1.0, 1.0]]])
    assert predict_forgetting_from_logits(pred_flip, gold, mask, vocab)[0] == 1
    pred_ok = np.array([[[9.0, 1.0, 1.0], [9.0, 1.0, 1.0]]])
    assert predict_forgetting_from_logits(pred_ok, gold, mask, vocab)[0] == 0
    # padded positions are ignored
    assert predict_forgetting_from_logits(pred_flip, gold, np.array([[True, False]]), vocab)[0] == 1


def test_margin_loss_sign_and_downweighting():
    correct = torch.tensor([2.0, -2.0])
    other = torch.tensor([0.0, 0.0])
    # z=0 (not forgotten): margin satisfied -> zero loss
    assert float(margin_loss(correct[:1], other[:1], torch.tensor([0.0]))) == 0.0
    # z=1 (forgotten): margin reversed -> zero loss as well
    assert float(margin_loss(correct[1:], other[1:], torch.tensor([1.0]))) == 0.0
    # violated margins produce positive loss
    assert float(margin_loss(torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]))) > 0
    # positives count with weight alpha = 0.1
    loss = margin_loss(
        torch.tensor([0.0, 0.0]), torch.tensor([2.0, 2.0]), torch.tensor([1.0, 0.0]),
        positive_weight=0.1,
    )
    # label 1 (forgotten) is satisfied (margin -1 -> 0 loss, weight 0.1);
    # label 0 violates the margin (1 + 2 = 3 loss, weight 1)
    expected = (0.1 * 0.0 + 1.0 * 3.0) / (0.1 + 1.0)
    assert abs(float(loss) - expected) < 1e-6


def test_sample_pairs_respects_allowed_upstream_and_has_both_classes():
    labels = np.zeros((4, 5), dtype=np.int8)
    labels[0, 0] = 1
    labels[1, 2] = 1
    allowed = np.array([True, True, True, False, False])
    rng = np.random.default_rng(0)
    pos, neg = sample_pairs(labels, 2, 2, rng, allowed_upstream=allowed)
    assert all(labels[i, j] == 1 for i, j in pos)
    assert all(labels[i, j] == 0 for i, j in neg)
    assert all(allowed[j] for _, j in np.concatenate([pos, neg], axis=0))


def test_frequency_prior_is_log_odds():
    labels = np.array([[1, 0], [1, 0], [0, 0], [0, 0]])
    prior = frequency_prior(labels)
    assert prior[1] < 0                 # never forgotten
    assert abs(prior[0] - np.log(0.5 / 0.5)) < 1e-6
