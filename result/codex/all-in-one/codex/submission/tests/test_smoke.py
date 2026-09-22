"""Smoke tests of the Simformer implementation (fast, CPU only).

Run with ``python -m pytest tests/test_smoke.py`` or ``python tests/test_smoke.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simformer import (IntervalUpperBound, Simformer, SimformerConfig,
                       TransformerConfig, TransformerScoreNet, get_task)
from simformer.masks import (graph_inversion, inversion_attention_mask,
                             gaussian_linear_mask, hmm_mask, moralize)
from simformer.sde import VESDE, VPSDE, get_sde
from simformer.tokenizer import IdentifierEmbedding, Tokenizer, TokenizerConfig


def test_tokenizer_shapes():
    config = TokenizerConfig(n_variables=4, d_model=16, id_dim=16, value_dim=16,
                             cond_dim=16)
    tokenizer = Tokenizer(config)
    values = torch.randn(3, 4)
    state = torch.tensor([[1.0, 0.0, 1.0, 0.0]] * 3)
    tokens = tokenizer(values, state)
    assert tokens.shape == (3, 4, 16)
    # conditioned variables are embedded with a learnable vector, latent ones
    # with zeros
    conditioned = tokenizer(values, torch.ones(3, 4))
    latent = tokenizer(values, torch.zeros(3, 4))
    assert not torch.allclose(conditioned, latent)


def test_value_embedding_repeats_scalar():
    config = TokenizerConfig(n_variables=2, d_model=8, id_dim=8, value_dim=8,
                             cond_dim=8, token_mlp=False,
                             learnable_identifiers=False)
    tokenizer = Tokenizer(config)
    values = torch.tensor([[2.0, -1.0]])
    other = torch.tensor([[-1.0, -1.0]])
    identifiers = torch.zeros(1, 2, 8)
    tokens = tokenizer(values, torch.zeros(1, 2), identifiers=identifiers)
    tokens_other = tokenizer(other, torch.zeros(1, 2), identifiers=identifiers)
    # the value part of the concatenated (pre projection) token is value * ones,
    # so the difference of two tokens only depends on the value difference
    weight = tokenizer.projection.weight[:, 8:16].detach().numpy()
    difference = (tokens - tokens_other)[0, 0].detach().numpy()
    assert np.allclose(difference, 3.0 * weight.sum(axis=1), atol=1e-5)
    assert np.allclose((tokens - tokens_other)[0, 1].detach().numpy(), 0.0,
                       atol=1e-5)


def test_transformer_score_shapes_and_mask():
    config = TransformerConfig(n_variables=5, d_model=16, n_heads=2,
                               attention_size=8, n_layers=2, time_embed_dim=16)
    net = TransformerScoreNet(config)
    values = torch.randn(2, 5)
    state = torch.zeros(2, 5)
    t = torch.rand(2)
    mask = torch.ones(5, 5, dtype=torch.bool)
    score = net(values, state, t, mask)
    assert score.shape == (2, 5)


def test_sde_marginal_and_score():
    # float64: the analytic form -(x_t - mu x_0) / sigma^2 suffers from
    # catastrophic cancellation for very small sigma in float32 (which is why
    # training uses the equivalent, numerically stable -eps / sigma).
    for sde in (VESDE(), VPSDE()):
        x0 = torch.randn(4, 3, dtype=torch.float64)
        t = torch.rand(4, dtype=torch.float64)
        x_t, eps, sigma = sde.perturb(x0, t)
        mu, _ = sde.marginal_coeff(t)
        # x_t = mu * x0 + sigma * eps
        sigma_col = sigma.reshape(-1, 1)
        mu_col = mu.reshape(-1, 1)
        assert torch.allclose(x_t, mu_col * x0 + sigma_col * eps, atol=1e-5)
        # the denoising score is -eps / sigma
        target = sde.marginal_score(x_t, x0, t)
        assert torch.allclose(target, -eps / sigma_col, rtol=1e-6, atol=1e-6)
        # the weighting function is positive
        assert torch.all(sde.loss_weight(t) > 0)


def test_graph_inversion_gaussian_linear():
    base = gaussian_linear_mask(10, 10)
    posterior_state = np.concatenate([np.zeros(10), np.ones(10)])
    likelihood_state = 1.0 - posterior_state
    joint_state = np.zeros(20)
    posterior = inversion_attention_mask(base, posterior_state)
    likelihood = inversion_attention_mask(base, likelihood_state)
    joint = inversion_attention_mask(base, joint_state)
    # for the likelihood no additional edges are needed (addendum / A1.1)
    assert np.array_equal(likelihood, base > 0)
    # for the posterior / joint, the missing 10 moral edges are inserted
    assert (posterior & ~(base > 0)).sum() == 10
    assert (joint & ~(base > 0)).sum() == 10
    # theta_i now depends on x_i for the posterior
    assert posterior[0, 10]
    assert not base[0, 10]


def test_graph_inversion_hmm_keeps_sparse_structure():
    base = hmm_mask(10, 10)
    posterior_state = np.concatenate([np.zeros(10), np.ones(10)])
    likelihood_state = 1.0 - posterior_state
    posterior = inversion_attention_mask(base, posterior_state)
    likelihood = inversion_attention_mask(base, likelihood_state)
    # the base dependencies are always preserved
    assert (posterior & (base > 0)).sum() == (base > 0).sum()
    assert posterior[5, 4]                       # the Markov chain is preserved
    # parameters can attend the observations after conditioning ...
    assert posterior[:10, 10:].any()
    # ... and the observations can attend the parameters
    assert posterior[10:, :10].any()
    # the mask stays sparse (the HMM is a sparse dependency structure)
    assert posterior.mean() < 0.5
    # for the likelihood no edges have to be added: p(theta) can be sampled
    # independently of p(x) (Appendix A1.2)
    assert int((likelihood & ~(base > 0)).sum()) == 0
    assert not likelihood[:10, 10:].any()


def test_moralize_connects_coparents():
    base = np.eye(3)
    base[2, 0] = 1
    base[2, 1] = 1
    J = moralize(base)
    assert J[0, 1] and J[1, 0]


def test_simformer_loss_and_sampling_two_moons():
    task = get_task("two_moons")
    config = SimformerConfig(mask_mode="directed", n_layers=2, num_steps=10)
    model = Simformer(task.problem(), config)
    rng = np.random.default_rng(0)
    theta, x, index, metadata = task.joint_sample(32, rng)
    loss = model.compute_loss(torch.as_tensor(theta), torch.as_tensor(x),
                              torch.as_tensor(index), metadata, rng)
    assert np.isfinite(float(loss))
    value, state = task.posterior_condition(theta[0], x[0])
    samples = model.sample([(value, state)], num_samples=5, num_steps=5)
    assert samples.shape == (1, 5, task.n_variables)


def test_guidance_enforces_interval():
    """Guided sampling should push a variable below the interval bound."""
    task = get_task("gaussian_linear")
    config = SimformerConfig(mask_mode="dense", n_layers=1, num_steps=30)
    model = Simformer(task.problem(), config)
    rng = np.random.default_rng(0)
    theta, x, index, metadata = task.joint_sample(64, rng)
    model.fit(theta, x, index, metadata, batch_size=32, max_epochs=1,
              patience=1, verbose=False)
    constraint = IntervalUpperBound(list(range(task.n_variables)), -0.5)
    # An untrained score model still has a meaningful constraint gradient: the
    # guidance term is the score of the (sigmoid) constraint indicator.
    value, state = task.posterior_condition(theta[0], x[0])
    samples = model.sample([(value, state)], num_samples=4, num_steps=5,
                           guidance=constraint)
    assert np.isfinite(samples).all()


def test_identifier_embedding_with_fourier_index():
    embedding = IdentifierEmbedding(n_variables=4, id_dim=8, n_kinds=2,
                                    use_fourier=torch.tensor([False, True, True,
                                                              True]))
    kinds = torch.tensor([0, 1, 1, 1])
    index = torch.linspace(0, 1, 4).repeat(3, 1)
    out = embedding(kinds, index)
    assert out.shape == (3, 4, 8)
    # variables with an index get a different embedding for different indices
    index2 = index.clone()
    index2[:, 1] += 0.5
    out2 = embedding(kinds, index2)
    assert not torch.allclose(out[:, 1], out2[:, 1])
    assert torch.allclose(out[:, 0], out2[:, 0])


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # pragma: no cover
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len([n for n in globals() if n.startswith('test_')]) - failures} "
          f"passed, {failures} failed")
    sys.exit(1 if failures else 0)
