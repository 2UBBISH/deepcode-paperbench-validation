"""Self-distillation tests (Section 4.4)."""

import torch
import torch.nn as nn

from apt.distill import (
    DistillWeights,
    IdentityLoRA,
    distill_loss,
    layer_mapping,
    layerwise_distillation_loss,
    sample_teacher_layers,
    total_loss,
)


def test_identity_lora_starts_as_identity():
    tr = IdentityLoRA(8, rank=4)
    x = torch.randn(2, 5, 8)
    torch.testing.assert_close(tr(x), x)


def test_identity_lora_can_learn_a_transform():
    tr = IdentityLoRA(6, rank=3)
    torch.nn.init.normal_(tr.lora_B, std=0.5)
    x = torch.randn(2, 4, 6)
    assert not torch.allclose(tr(x), x, atol=1e-6)


def test_distillation_weights_per_task():
    glue = DistillWeights.for_task("sst2")
    assert (glue.pred_weight, glue.layer_weight) == (1.0, 0.9)
    squad = DistillWeights.for_task("squad")
    assert (squad.pred_weight, squad.layer_weight) == (0.1, 0.9)
    cnn = DistillWeights.for_task("cnn_dm")
    assert (cnn.pred_weight, cnn.layer_weight) == (0.1, 0.9)


def test_total_loss_interpolates_with_mu():
    dist = torch.tensor(1.0)
    ft = torch.tensor(3.0)
    assert torch.allclose(total_loss(dist, ft, 0.0), ft)
    assert torch.allclose(total_loss(dist, ft, 1.0), dist)
    assert torch.allclose(total_loss(dist, ft, 0.5), torch.tensor(2.0))


def test_layer_mapping_and_sampling():
    sampled = sample_teacher_layers(12, 5, generator=torch.Generator().manual_seed(0))
    assert len(sampled) == 5 and sampled == sorted(sampled)
    mapping = layer_mapping([1, 2, 3], [1, 2, 3])
    assert set(mapping) == {1, 2, 3}


def test_layerwise_loss_is_zero_for_identical_hidden_states():
    tr = IdentityLoRA(4, rank=2)
    h = [torch.randn(2, 3, 4) for _ in range(4)]
    loss = layerwise_distillation_loss(tr, h, h, {1: 1, 2: 2})
    assert float(loss) < 1e-6
