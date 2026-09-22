"""Adaptive tuning tests (Section 4.3)."""

import torch

from apt.tuning import adapter_importances, salient_adapter_names


class DummyAdapter:
    def __init__(self, importance: float, rank: int = 4, use_lora: bool = True):
        self.use_lora = use_lora
        self.rank = rank
        self._importance = importance

    def adapter_importance(self):
        return self._importance


class DummyTopo:
    def __init__(self, linears):
        self.linears = linears


def test_salient_adapters_take_the_top_half():
    topo = DummyTopo({f"l{i}": DummyAdapter(float(i)) for i in range(6)})
    names = salient_adapter_names(topo, top_fraction=0.5)
    assert set(names) == {"l3", "l4", "l5"}
    scores = adapter_importances(topo)
    assert scores.shape == (6,)


def test_adapters_without_lora_are_ignored():
    topo = DummyTopo(
        {
            "a": DummyAdapter(1.0),
            "b": DummyAdapter(2.0, use_lora=False),
            "c": DummyAdapter(3.0),
        }
    )
    assert set(salient_adapter_names(topo, 0.5)) == {"c"}
