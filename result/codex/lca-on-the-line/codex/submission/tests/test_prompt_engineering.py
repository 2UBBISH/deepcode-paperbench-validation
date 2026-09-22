"""Tests for the taxonomy prompts of Section 4.3.3 / Table 14."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lca_on_the_line.hierarchy import load_wordnet_hierarchy  # noqa: E402
from lca_on_the_line.prompt_engineering import (  # noqa: E402
    TAXONOMY_PROMPT_MODES,
    build_taxonomy_prompts,
    class_prompts_for_ensemble,
    imagenet_classnames,
    taxonomy_parent_names,
)


@pytest.fixture(scope="module")
def hierarchy():
    return load_wordnet_hierarchy()


def test_dalmatian_taxonomy_path(hierarchy):
    names = imagenet_classnames()
    index = names.index("dalmatian")
    parents = taxonomy_parent_names(hierarchy, depth=2)[index]
    assert parents == ["dog", "canine"]
    prompts = build_taxonomy_prompts(hierarchy, "taxonomy_parent", depth=2)
    assert prompts[index] == (
        "a photo of a dalmatian, which is a type of a dog, "
        "which is a type of a canine."
    )


def test_prompt_modes(hierarchy):
    names = imagenet_classnames()
    index = names.index("dalmatian")
    prompts = {
        mode: build_taxonomy_prompts(hierarchy, mode, depth=2)[index]
        for mode in TAXONOMY_PROMPT_MODES
    }
    assert prompts["baseline"] == "a photo of a dalmatian."
    assert "which is a type of" not in prompts["baseline"]
    assert "dalmatian, dog, canine" in prompts["stack_parent"]
    assert "which is a type of" not in prompts["stack_parent"]
    assert "which is a type of" in prompts["taxonomy_parent"]
    assert "which is a type of" in prompts["shuffle_parent"]
    # the shuffled ancestors must not be the true ancestors
    assert "a dog" not in prompts["shuffle_parent"]
    assert "a canine" not in prompts["shuffle_parent"]


def test_shuffle_is_deterministic_and_differs_from_true_path(hierarchy):
    names = imagenet_classnames()
    index = names.index("dalmatian")
    a = build_taxonomy_prompts(hierarchy, "shuffle_parent", depth=2, seed=0)[index]
    b = build_taxonomy_prompts(hierarchy, "shuffle_parent", depth=2, seed=0)[index]
    c = build_taxonomy_prompts(hierarchy, "shuffle_parent", depth=2, seed=1)[index]
    assert a == b
    assert a != c


def test_ensemble_transpose(hierarchy):
    ensemble = class_prompts_for_ensemble(
        hierarchy, mode="baseline", templates=["a photo of a {}.", "a {}."]
    )
    assert len(ensemble) == 1000
    assert len(ensemble[0]) == 2


def test_evaluate_prompt_modes_with_a_stub_classifier(monkeypatch, hierarchy):
    """The Table-14 harness runs end-to-end with a stub zero-shot model."""
    from lca_on_the_line import experiments_prompt as ep
    from lca_on_the_line.data import IndexedImageDataset

    class StubClassifier:
        def __init__(self):
            self.calls = []

        def encode_prompts(self, class_prompts):
            self.calls.append(len(class_prompts))
            # deterministic embedding per prompt string
            vecs = []
            for prompts in class_prompts:
                seed = abs(hash(prompts[0])) % (2 ** 31)
                rng = np.random.RandomState(seed)
                vecs.append(rng.randn(8))
            v = torch.tensor(np.array(vecs), dtype=torch.float32)
            return v / v.norm(dim=-1, keepdim=True)

        def logits_with_text(self, images, text_features):
            n = len(images)
            return np.random.RandomState(0).randn(n, text_features.shape[0])

    stub = StubClassifier()
    monkeypatch.setattr(ep, "build_classifier", lambda *a, **k: stub)

    targets = [0, 1, 2, 3]
    dataset = IndexedImageDataset(
        [(None, t) for t in targets], name="stub"
    )

    class StubImage:
        def __init__(self, t):
            self.t = t

    monkeypatch.setattr(
        IndexedImageDataset, "__getitem__",
        lambda self, i: (StubImage(self.samples[i][1]), self.samples[i][1]),
    )
    monkeypatch.setattr(
        ep, "load_dataset_by_name", lambda name, root, **kw: dataset
    )
    results = ep.evaluate_prompt_modes(
        "CLIP_ViT-B_32",
        data_root="/tmp",
        datasets=["imagenet"],
        modes=list(TAXONOMY_PROMPT_MODES),
        limit=4,
        hierarchy=hierarchy,
    )
    assert set(results["imagenet"]) == set(TAXONOMY_PROMPT_MODES)
    for metrics in results["imagenet"].values():
        assert 0.0 <= metrics["top1"] <= 1.0
        assert metrics["ce"] >= 0.0
    assert stub.calls == [1000] * len(TAXONOMY_PROMPT_MODES)
