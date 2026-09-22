"""Tests for the baseline implementations."""

import torch

from apt.wrap import WrapConfig, compute_block_salience, wrap_model
from baselines.lora import LoRALinear, attach_lora, merge_and_unwrap
from baselines.mask_tuning import mask_search
from tests.test_wrap import tiny_roberta


def test_attach_lora_covers_query_and_value():
    model = tiny_roberta()
    wrapped = attach_lora(model, r=4)
    assert wrapped, "no projection was wrapped"
    names = {attr for _, attr, _ in wrapped}
    assert names == {"query", "value"}
    assert all(isinstance(lin, LoRALinear) for _, _, lin in wrapped)


def test_merge_and_unwrap_restores_plain_linears():
    import torch.nn as nn

    model = tiny_roberta()
    wrapped = attach_lora(model, r=4)
    torch.nn.init.normal_(wrapped[0][2].lora_A, std=0.1)
    torch.nn.init.normal_(wrapped[0][2].lora_B, std=0.1)
    base_before = wrapped[0][2].base.weight.clone()
    merge_and_unwrap(wrapped)
    for parent, attr, lin in wrapped:
        assert isinstance(getattr(parent, attr), nn.Linear)
        assert not isinstance(getattr(parent, attr), LoRALinear)
    assert not torch.allclose(base_before, wrapped[0][2].base.weight)


def test_mask_search_reaches_the_target_sparsity():
    torch.manual_seed(0)
    model = tiny_roberta()
    topo = wrap_model(model, WrapConfig())
    topo.freeze_backbone()
    ids = torch.randint(0, 128, (4, 10))
    out = model(input_ids=ids, labels=torch.tensor([0, 1, 0, 1]))
    out.loss.backward()
    importance = compute_block_salience(topo, use_kurtosis=True)
    mask = mask_search(topo, importance, target_sparsity=0.5)
    state = topo.build_pruning_state()
    state.retain = mask
    assert state.current_sparsity() >= 0.5 - 1e-6
    # dimensions are never pruned by the baseline
    assert all(mask[i] for i, b in enumerate(topo.blocks) if b.kind == 2)
