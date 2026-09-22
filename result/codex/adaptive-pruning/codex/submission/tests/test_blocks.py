"""Tests for the block table, the salience score and the mask search."""

import torch

from apt.blocks import Block, DIM, HEAD, NEURON, PruningState, kurtosis_from_moments


def test_appendix_c_block_costs():
    """The numbers quoted in Appendix C for RoBERTa-base."""
    d_m, n_h, n_f, n_L = 768, 12, 3072, 12
    d_h = d_m // n_h
    assert 4 * d_m * d_m // n_h == 196608      # a head
    assert 2 * d_m == 1536                     # a neuron
    assert n_L * (4 * d_m + 2 * n_f) == 110592  # a hidden dimension


def test_pruning_state_hits_the_target_sparsity():
    blocks = []
    for i in range(8):
        blocks.append(Block(f"head{i}", HEAD, "g", param_count=100))
    for i in range(8):
        blocks.append(Block(f"neuron{i}", NEURON, "g", param_count=10))
    for i in range(4):
        blocks.append(Block(f"dim{i}", DIM, "g", param_count=1000))
    state = PruningState(blocks, d_h=8)
    assert state.total_parameters == 4 * (4 * 8 * 8 + 2 * 8)
    state.salience_ema = torch.arange(len(blocks), dtype=torch.float64)
    for target in (0.2, 0.5, 0.8):
        state.select_for_budget(1.0 - target)
        assert state.current_sparsity() >= target - 1e-9


def test_prefix_cost_is_monotone_and_matches_full_model():
    blocks = [Block(f"head{i}", HEAD, "g", param_count=64) for i in range(6)]
    blocks += [Block(f"neuron{i}", NEURON, "g", param_count=8) for i in range(6)]
    blocks += [Block(f"dim{i}", DIM, "g", param_count=100) for i in range(3)]
    state = PruningState(blocks, d_h=8)
    order = torch.randperm(len(blocks), generator=torch.Generator().manual_seed(0))
    costs = state._prefix_costs(order)
    assert costs.numel() == len(blocks) + 1
    assert bool((costs[1:] >= costs[:-1]).all())
    assert abs(float(costs[-1]) - state.total_parameters) < 1e-6


def test_mask_annealing_moves_slowly():
    blocks = [Block(f"head{i}", HEAD, "g", param_count=64) for i in range(4)]
    blocks += [Block(f"dim{i}", DIM, "g", param_count=100) for i in range(2)]
    state = PruningState(blocks, d_h=8, alpha=0.01)
    assert float(state.mask.min()) == 1.0        # starts dense
    state.salience_ema = torch.arange(len(blocks), dtype=torch.float64)
    state.select_for_budget(0.6)
    state.anneal_masks()
    pruned = ~state.retain
    assert float(state.mask[pruned].max()) <= 0.99 + 1e-9
    assert float(state.mask[~pruned].min()) == 1.0
    for _ in range(200):
        state.anneal_masks()
    assert abs(float(state.mask[pruned].max()) - 0.0) < 1e-9
    assert abs(float(state.mask.max()) - 1.0) < 1e-9


def test_kurtosis_matches_scipy():
    torch.manual_seed(0)
    x = torch.randn(500, 7) * 2 + 1
    moments = torch.stack([x.sum(0), x.pow(2).sum(0), x.pow(3).sum(0), x.pow(4).sum(0)]).double()
    got = kurtosis_from_moments(moments, x.shape[0])
    try:
        from scipy.stats import kurtosis as sp_kurt

        expected = torch.tensor(sp_kurt(x.numpy(), fisher=False, bias=True)).double()
        # moments are accumulated in float32 on-device then upcast to float64
        assert torch.allclose(got, expected, rtol=1e-3, atol=1e-3)
    except ImportError:  # pragma: no cover
        assert torch.isfinite(got).all()
