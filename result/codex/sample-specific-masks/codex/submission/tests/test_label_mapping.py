"""Tests of the output mappings (Section 2.3, Appendix A.4, Algorithms 2-4)."""

import torch

from smm.label_mapping import (
    LabelMapping,
    frequency_distribution,
    frequent_label_mapping,
    greedy_injective_mapping,
    random_label_mapping,
)


def test_frequency_distribution_counts():
    preds = [torch.tensor([0, 1, 1]), torch.tensor([2, 1, 0])]
    labels = [torch.tensor([0, 0, 1]), torch.tensor([2, 1, 0])]
    d = frequency_distribution(preds, labels, num_pretrained_classes=4, num_target_classes=3)
    assert d.shape == (4, 3)
    # pairs: (0,0) (1,0) (1,1) (2,2) (1,1) (0,0)
    assert int(d[1, 1]) == 2
    assert int(d[0, 0]) == 2
    assert int(d[1, 0]) == 1
    assert int(d.sum()) == 6


def test_greedy_mapping_is_injective_and_uses_the_largest_counts():
    d = torch.zeros(5, 3, dtype=torch.long)
    d[2, 0] = 10
    d[3, 1] = 9
    d[4, 2] = 8
    mapping = greedy_injective_mapping(d, 3)
    assert mapping.source.tolist() == [2, 3, 4]
    assert mapping.target.tolist() == [0, 1, 2]
    assert len(set(mapping.source.tolist())) == 3


def test_greedy_mapping_handles_missing_classes():
    d = torch.zeros(4, 3, dtype=torch.long)
    d[0, 0] = 5
    mapping = greedy_injective_mapping(d, 3)
    assert len(set(mapping.source.tolist())) == 3
    assert len(set(mapping.target.tolist())) == 3


def test_random_label_mapping():
    a = random_label_mapping(1000, 10, seed=0)
    b = random_label_mapping(1000, 10, seed=0)
    c = random_label_mapping(1000, 10, seed=1)
    assert a.source.tolist() == b.source.tolist()
    assert a.source.tolist() != c.source.tolist()
    assert len(set(a.source.tolist())) == 10


def test_frequent_label_mapping_uses_the_identity_input():
    # Every image of target class 0 is predicted as pre-trained class 7, etc.
    preds = [torch.tensor([7, 7, 9, 9])]
    labels = [torch.tensor([0, 0, 1, 1])]
    mapping = frequent_label_mapping(preds, labels, 10, 2)
    assert dict(zip(mapping.source.tolist(), mapping.target.tolist())) == {7: 0, 9: 1}


def test_target_index_of_and_serialisation():
    mapping = LabelMapping(torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]), 10, 7, "ilm")
    idx = mapping.target_index_of(torch.tensor([5, 6, 4]))
    assert idx.tolist() == [1, 2, 0]
    restored = LabelMapping.from_dict(mapping.as_dict())
    assert restored.source.tolist() == mapping.source.tolist()
    assert restored.target.tolist() == mapping.target.tolist()
