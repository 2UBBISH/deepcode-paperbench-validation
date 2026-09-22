"""CIDEr against the reference implementation.

The captioning numbers of Table 1 come from ``pycocoevalcap``'s CIDEr(-D).  The
implementation in :mod:`robust_clip.eval.cider` is self-contained (no Java
tokenizer, no extra dependency), so it is pinned against the reference: with
``scale=10`` it must reproduce ``pycocoevalcap``'s value exactly.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robust_clip.eval.cider import CiderScorer  # noqa: E402


def _corpus(seed: int = 0, n: int = 40, n_refs: int = 5):
    vocab = [
        "cat", "dog", "mat", "park", "train", "station", "runs", "sits", "on",
        "the", "a", "green", "red", "bicycle", "river",
    ]
    rng = np.random.RandomState(seed)
    gts, res = {}, {}
    for i in range(n):
        references = [" ".join(rng.choice(vocab, size=8)) for _ in range(n_refs)]
        gts[i] = references
        if i % 2 == 0:
            candidate = " ".join(references[0].split()[:6] + ["extra"])
        else:
            candidate = " ".join(rng.choice(vocab, size=8))
        res[i] = [candidate]
    return gts, res


def test_cider_matches_pycocoevalcap():
    reference = pytest.importorskip("pycocoevalcap.cider.cider")
    gts, res = _corpus()
    official, official_scores = reference.Cider().compute_score(gts, res)

    references = [gts[i] for i in range(len(gts))]
    candidates = [res[i][0] for i in range(len(res))]
    mine = CiderScorer(scale=10.0).prepare(references).compute_scores(candidates, references)

    assert np.allclose(np.array(mine), np.array(official_scores), atol=1e-6)
    assert abs(float(np.mean(mine)) - float(official)) < 1e-6


def test_cider_default_scale_is_percent():
    gts, res = _corpus(seed=1)
    references = [gts[i] for i in range(len(gts))]
    candidates = [res[i][0] for i in range(len(res))]
    percent = CiderScorer().prepare(references).compute_scores(candidates, references)
    official = CiderScorer(scale=10.0).prepare(references).compute_scores(candidates, references)
    assert np.allclose(np.array(percent), 100.0 * np.array(official), atol=1e-6)


def test_cider_ranks_a_matching_caption_above_an_unrelated_one():
    references = [
        ["a cat is sitting on a mat", "the cat sits on the mat", "a cat rests on a mat"],
        ["a dog runs in the park", "a dog is running outside", "a brown dog runs"],
    ]
    scorer = CiderScorer().prepare(references)
    good = scorer.score("a cat is sitting on a mat", references[0])
    bad = scorer.score("a train is leaving the station", references[0])
    assert good > bad
    assert good > 0.0


if __name__ == "__main__":  # pragma: no cover
    test_cider_matches_pycocoevalcap()
    test_cider_default_scale_is_percent()
    test_cider_ranks_a_matching_caption_above_an_unrelated_one()
    print("cider tests OK")
