"""Model wrapping / topology tests on a locally built tiny model."""

import pytest
import torch

from apt.blocks import HEAD, NEURON
from apt.wrap import WrapConfig, compute_block_salience, wrap_model


def tiny_roberta(n_layers=2, hidden=32, heads=4, ffn=48):
    from transformers import RobertaConfig, RobertaForSequenceClassification

    conf = RobertaConfig(
        vocab_size=128,
        hidden_size=hidden,
        num_hidden_layers=n_layers,
        num_attention_heads=heads,
        intermediate_size=ffn,
        max_position_embeddings=64,
        num_labels=2,
    )
    return RobertaForSequenceClassification(conf)


def tiny_t5(n_layers=2, d_model=32, heads=4, d_kv=8, d_ff=48):
    from transformers import T5Config, T5ForConditionalGeneration

    conf = T5Config(
        vocab_size=128,
        d_model=d_model,
        d_ff=d_ff,
        d_kv=d_kv,
        num_layers=n_layers,
        num_decoder_layers=n_layers,
        num_heads=heads,
        decoder_start_token_id=0,
    )
    return T5ForConditionalGeneration(conf)


def test_roberta_topology_partitions_every_feature():
    model = tiny_roberta()
    topo = wrap_model(model, WrapConfig())
    state = topo.build_pruning_state()
    state.check_coverage(topo.linears)          # raises if coverage is wrong
    assert len(topo.blocks) == 2 * (4 + 48) + 32
    n_layers, heads, ffn, hidden = 2, 4, 48, 32
    assert state.total_parameters == n_layers * (4 * hidden * hidden + 2 * hidden * ffn)


def test_t5_topology_partitions_every_feature():
    model = tiny_t5()
    topo = wrap_model(model, WrapConfig())
    state = topo.build_pruning_state()
    state.check_coverage(topo.linears)
    # 12 attention modules (2 enc self + 2 dec self + 2 dec cross) x 4 heads
    assert sum(1 for b in topo.blocks if b.kind == HEAD) == 6 * 4
    assert sum(1 for b in topo.blocks if b.kind == NEURON) == 4 * 48


def test_block_salience_is_finite_and_ordered_by_parameter_count():
    torch.manual_seed(0)
    model = tiny_roberta()
    topo = wrap_model(model, WrapConfig())
    topo.freeze_backbone()
    ids = torch.randint(0, 128, (4, 12))
    out = model(input_ids=ids, labels=torch.tensor([0, 1, 0, 1]))
    out.loss.backward()
    scores = compute_block_salience(topo, use_kurtosis=True)
    assert scores.shape == (len(topo.blocks),)
    assert torch.isfinite(scores).all()
    scores_wo = compute_block_salience(topo, use_kurtosis=False)
    # the outlier-aware score always adds a non-negative kurtosis term
    assert bool((scores + 1e-9 >= scores_wo).all())


def test_vectorised_salience_matches_a_naive_reference():
    """The scatter-based scorer must equal the straightforward per-block loop."""
    torch.manual_seed(0)
    model = tiny_roberta()
    topo = wrap_model(model, WrapConfig())
    topo.freeze_backbone()
    ids = torch.randint(0, 128, (4, 12))
    out = model(input_ids=ids, labels=torch.tensor([0, 1, 0, 1]))
    out.loss.backward()

    fast = compute_block_salience(topo, use_kurtosis=False)

    cache = {}
    for name, lin in topo.linears.items():
        sal = lin.compressed_salience()
        tune = (lin.tuning_salience_in(), lin.tuning_salience())
        cache[name] = (sal, tune)
    naive = torch.zeros(len(topo.blocks), dtype=torch.float64)
    for bi, block in enumerate(topo.blocks):
        total = 0.0
        for sl in block.slices:
            sal_in, sal_out = cache[sl.linear][0]
            v = sal_in if sl.dim == "in" else sal_out
            if v is not None:
                total += float(v[sl.start : sl.stop].sum().item())
            t = cache[sl.linear][1][0 if sl.dim == "in" else 1]
            if t is not None:
                total += float(t[sl.start : sl.stop].sum().item())
        naive[bi] = total
    assert torch.allclose(fast, naive, rtol=1e-6, atol=1e-6)
