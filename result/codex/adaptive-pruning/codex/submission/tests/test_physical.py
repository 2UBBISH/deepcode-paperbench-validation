"""Physical pruning tests: the shrunk model must still run and be smaller."""

import torch

from apt.physical import materialize, n_parameters
from apt.wrap import WrapConfig, compute_block_salience, wrap_model
from tests.test_wrap import tiny_roberta, tiny_t5


def _prune(topo, model, ids, labels=None):
    topo.freeze_backbone()
    state = topo.build_pruning_state()
    out = model(input_ids=ids, labels=labels if labels is not None else ids)
    out.loss.backward()
    state.update_salience(compute_block_salience(topo, use_kurtosis=True))
    state.select_for_budget(0.4)
    state.harden_masks()
    state.write_masks_to_linears(topo.linears)
    return state


def test_roberta_physical_pruning_runs_and_shrinks():
    torch.manual_seed(0)
    model = tiny_roberta()
    topo = wrap_model(model, WrapConfig())
    ids = torch.randint(0, 128, (3, 10))
    state = _prune(topo, model, ids, labels=torch.tensor([0, 1, 0]))
    small = materialize(topo, state)
    small.eval()
    with torch.no_grad():
        out = small(input_ids=ids)
    assert out.logits.shape == (3, 2)
    assert n_parameters(small) < n_parameters(model)
    assert small.roberta.embeddings.word_embeddings.weight.shape[1] == len(
        [b for i, b in enumerate(topo.blocks) if b.kind == 2 and bool(state.retain[i])]
    )


def test_t5_physical_pruning_runs_and_shrinks():
    torch.manual_seed(0)
    model = tiny_t5()
    topo = wrap_model(model, WrapConfig())
    ids = torch.randint(0, 128, (3, 9))
    state = _prune(topo, model, ids)
    small = materialize(topo, state)
    small.eval()
    with torch.no_grad():
        out = small(input_ids=ids, labels=ids)
    assert torch.isfinite(out.loss)
    assert n_parameters(small) < n_parameters(model)
    enc_heads = {m.n_heads for m in [b.layer[0].SelfAttention for b in small.encoder.block]}
    assert len(enc_heads) == 1        # relative position bias forces uniformity
