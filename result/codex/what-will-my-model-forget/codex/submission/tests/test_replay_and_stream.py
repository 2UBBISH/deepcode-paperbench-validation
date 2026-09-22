import numpy as np
import pytest

from wwmf.data.types import Example
from wwmf.forecasting.types import (
    CandidateVocab,
    OnlineArtifact,
    UpstreamCache,
    UpstreamExampleCache,
)
from wwmf.refinement.replay import GTForgetReplay, RandomReplay, ReplayPool, ScoreReplay
from wwmf.refinement.stream import shuffle_stream


def _cache(n=8, T=3, C=4) -> UpstreamCache:
    items = [
        UpstreamExampleCache(
            example=Example(f"in{j}", f"out{j}", task="t", idx=j),
            reduced_logits=np.full((T, C), float(j), dtype=np.float16),
            gold_ids=np.zeros(T, dtype=np.int64),
            mask=np.ones(T, dtype=bool),
            decoder_reps=np.zeros((T, 2), dtype=np.float16),
            pooled_rep=np.zeros(2, dtype=np.float16),
            correct=True,
        )
        for j in range(n)
    ]
    return UpstreamCache(
        examples=[it.example for it in items],
        vocab=CandidateVocab(np.arange(C, dtype=np.int64), 2, C),
        items=items,
    )


class _FakeForecaster:
    """Predicts the first three upstream examples as forgotten."""

    def predict(self, artifact):
        labels = np.zeros(len(artifact.labels), dtype=int)
        labels[:3] = 1
        return labels


def test_random_replay_returns_unique_indices_of_requested_size():
    pool = ReplayPool(_cache(10), max_target_len=3)
    strategy = RandomReplay(seed=0)
    picked = strategy.select(4, pool)
    assert len(picked) == 4
    assert len(set(picked)) == 4
    assert all(0 <= i < 10 for i in picked)


def test_score_replay_only_replays_predicted_forgotten_examples():
    cache = _cache(10)
    pool = ReplayPool(cache, max_target_len=3)
    artifact = OnlineArtifact(
        example=Example("x", "y", task="r"),
        delta_logits=np.zeros((3, 4), dtype=np.float16),
        gold_ids=np.zeros(3, dtype=np.int64),
        mask=np.ones(3, dtype=bool),
        labels=np.zeros(10, dtype=np.int8),
    )
    strategy = ScoreReplay(_FakeForecaster(), lambda ex: artifact, seed=0)
    picked = strategy.select(2, pool, online_example=artifact.example)
    assert set(picked).issubset({0, 1, 2})


def test_gt_forget_replay_uses_ground_truth_labels():
    cache = _cache(6)
    pool = ReplayPool(cache, max_target_len=3)
    labels = np.array([0, 0, 1, 1, 0, 0], dtype=np.int8)
    strategy = GTForgetReplay(lambda ex: labels, seed=0)
    picked = strategy.select(2, pool, online_example=Example("x", "y"))
    assert set(picked) == {2, 3}


def test_replay_batch_has_cached_logits():
    pool = ReplayPool(_cache(5, T=3, C=4), max_target_len=3)
    batch = pool.batch([0, 2])
    assert batch.logits.shape == (2, 3, 4)
    assert batch.inputs == ["in0", "in2"]
    assert batch.targets == ["out0", "out2"]


def test_shuffle_stream_is_deterministic_and_permutes():
    examples = [Example(str(i), str(i), idx=i) for i in range(10)]
    a = shuffle_stream(examples, seed=0)
    b = shuffle_stream(examples, seed=0)
    assert [e.idx for e in a] == [e.idx for e in b]
    assert sorted(e.idx for e in a) == list(range(10))
