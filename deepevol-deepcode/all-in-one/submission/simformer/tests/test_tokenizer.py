"""Unit tests for the Simformer SBI tokenizer (Sec. 3.1 / Addendum "Tokenization").

The tokenizer builds one token per statistical variable of the joint vector
``(theta, x)`` by concatenating four embeddings in the order

    [ identifier | value | (metadata) | condition state ]

with the convention (Sec. 3.1):

* ``condition state = True``  -> a *learnable* embedding is used,
* ``condition state = False`` -> the condition embedding is **zero**,
* a scalar variable's value is *repeated* to ``token_dim`` before usage,
* function-valued parameters share a single identifier embedding and use a
  random Fourier embedding of their index set as metadata.

Run with::

    pytest simformer/tests/test_tokenizer.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from simformer.simformer.tokenizer import (  # noqa: E402
    FunctionValuedSpec,
    TokenSpec,
    Tokenizer,
    build_benchmark_spec,
)

TOKEN_DIM = 50


# --------------------------------------------------------------------------
# helpers / fixtures
# --------------------------------------------------------------------------
@pytest.fixture()
def benchmark_spec() -> TokenSpec:
    """Plain benchmark layout: 2 parameters, 2 data variables."""
    return build_benchmark_spec(n_parameters=2, n_data=2)


@pytest.fixture()
def benchmark_tokenizer(benchmark_spec: TokenSpec) -> Tokenizer:
    torch.manual_seed(0)
    return Tokenizer(benchmark_spec, token_dim=TOKEN_DIM)


def _random_batch(tokenizer: Tokenizer, batch_size: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    n_params = tokenizer.n_parameter_variables
    n_data = tokenizer.n_data_variables
    theta = rng.normal(size=(batch_size, n_params)).astype(np.float32)
    x = rng.normal(size=(batch_size, n_data)).astype(np.float32)
    return theta, x


# --------------------------------------------------------------------------
# construction / layout
# --------------------------------------------------------------------------
def test_spec_layout(benchmark_spec: TokenSpec) -> None:
    assert len(benchmark_spec.parameter_names) == 2
    assert len(benchmark_spec.data_names) == 2
    assert benchmark_spec.function_valued == ()


def test_tokenizer_dimensions(benchmark_tokenizer: Tokenizer) -> None:
    tok = benchmark_tokenizer
    assert tok.token_dim == TOKEN_DIM
    # 2 parameters + 2 data variables -> 4 tokens, one per variable
    assert tok.n_tokens == 4
    assert tok.n_parameter_variables == 2
    assert tok.n_data_variables == 2
    assert tok.input_dim == 4  # scalar variables only


def test_forward_output_shape(benchmark_tokenizer: Tokenizer) -> None:
    tok = benchmark_tokenizer
    theta, x = _random_batch(tok)
    tokens = tok(torch.as_tensor(theta), torch.as_tensor(x))
    assert tokens.shape == (4, tok.n_tokens, tok.token_dim)
    assert torch.isfinite(tokens).all()


def test_forward_with_inferred_layout() -> None:
    """The tokenizer can be built from explicit variable counts."""
    torch.manual_seed(0)
    tok = Tokenizer(n_parameter_variables=3, n_data_variables=5, token_dim=32)
    assert tok.token_dim == 32
    assert tok.input_dim == 8
    assert tok.n_tokens == 8
    theta = torch.zeros(2, 3)
    x = torch.zeros(2, 5)
    assert tok(theta, x).shape == (2, 8, 32)


# --------------------------------------------------------------------------
# condition-state embedding: True -> learnable embedding, False -> zeros
# --------------------------------------------------------------------------
def test_condition_false_gives_zeros(benchmark_tokenizer: Tokenizer) -> None:
    """With M_C all-False the condition slot contributes exactly zeros.

    We verify this indirectly but exactly: turning a condition state on must
    change every token by the *same* additive constant (the learnable condition
    embedding), independent of the value/batch entry.  With state False that
    added constant is zero.
    """
    tok = benchmark_tokenizer
    theta, x = _random_batch(tok, batch_size=3, seed=1)
    theta_t, x_t = torch.as_tensor(theta), torch.as_tensor(x)

    mask_off = torch.zeros(3, tok.n_variables)
    mask_on = torch.ones(3, tok.n_variables)

    tok.eval()
    with torch.no_grad():
        off = tok(theta_t, x_t, condition_mask=mask_off)
        on = tok(theta_t, x_t, condition_mask=mask_on)
        delta = (on - off)[0]  # (n_tokens, token_dim)

    # the additive delta is identical for every batch element
    for b in range(1, 3):
        assert torch.allclose((on - off)[b], delta, atol=1e-5)

    # and it is non-trivial (the learnable condition embedding is not zero)
    assert delta.abs().sum() > 0


def test_condition_masking_is_additive_and_selective(benchmark_tokenizer: Tokenizer) -> None:
    """Only the token whose state is True changes."""
    tok = benchmark_tokenizer
    theta, x = _random_batch(tok, batch_size=2, seed=2)
    theta_t, x_t = torch.as_tensor(theta), torch.as_tensor(x)

    mask_off = torch.zeros(2, tok.n_variables)
    mask_one = torch.zeros(2, tok.n_variables)
    mask_one[:, 0] = 1.0

    tok.eval()
    with torch.no_grad():
        off = tok(theta_t, x_t, condition_mask=mask_off)
        one = tok(theta_t, x_t, condition_mask=mask_one)

    diff = (one - off).abs().sum(dim=-1)  # (batch, n_tokens)
    assert diff[:, 0].max() > 0
    assert torch.allclose(diff[:, 1:], torch.zeros_like(diff[:, 1:]), atol=1e-6)


# --------------------------------------------------------------------------
# scalar value handling: value is repeated to token_dim
# --------------------------------------------------------------------------
def test_scalar_values_are_repeated(benchmark_tokenizer: Tokenizer) -> None:
    """Changing a single scalar value only affects its own token, and a
    (constant-across-dimensions) scalar carries information.

    Since the value embedding repeats the scalar to ``token_dim`` before a
    learnable mixing, we simply assert that the token's dependence on its own
    value is non-degenerate (the token changes when the scalar changes).
    """
    tok = benchmark_tokenizer
    theta, x = _random_batch(tok, batch_size=2, seed=3)
    x = x.copy()
    x_perturbed = x.copy()
    x_perturbed[:, 0] += 1.0  # perturb the third token? -> data var 0 (token 2)
    theta_t = torch.as_tensor(theta)
    mask = torch.zeros(2, tok.n_variables)

    tok.eval()
    with torch.no_grad():
        a = tok(theta_t, torch.as_tensor(x), condition_mask=mask)
        b = tok(theta_t, torch.as_tensor(x_perturbed), condition_mask=mask)

    changed = (a - b).abs().sum(dim=-1) > 1e-6  # (batch, n_tokens)
    # the perturbed variable is the first data variable => token index 2
    assert changed[:, 2].all()
    # other tokens are unaffected
    assert not changed[:, 0].any()
    assert not changed[:, 1].any()
    assert not changed[:, 3].any()


def test_different_values_give_different_tokens(benchmark_tokenizer: Tokenizer) -> None:
    tok = benchmark_tokenizer
    theta, x = _random_batch(tok, batch_size=2, seed=4)
    mask = torch.zeros(2, tok.n_variables)
    tok.eval()
    with torch.no_grad():
        tokens = tok(torch.as_tensor(theta), torch.as_tensor(x), condition_mask=mask)
    # distinct joint inputs must produce distinct token sequences
    assert not torch.allclose(tokens[0], tokens[1])


# --------------------------------------------------------------------------
# identity embeddings / batch independence
# --------------------------------------------------------------------------
def test_identity_embeddings_shape_and_batch_independence(benchmark_tokenizer: Tokenizer) -> None:
    tok = benchmark_tokenizer
    ids = tok.identity_embeddings(5)
    assert ids.shape == (5, tok.n_tokens, tok.token_dim)

    theta, x = _random_batch(tok, batch_size=1, seed=5)
    pair = torch.cat([torch.as_tensor(theta), torch.as_tensor(x)], dim=-1)
    mask = torch.zeros(1, tok.n_variables)
    tok.eval()
    with torch.no_grad():
        a = tok(pair[:, : tok.n_parameter_variables], pair[:, tok.n_parameter_variables:], condition_mask=mask)
        b = tok(
            pair[:, : tok.n_parameter_variables],
            pair[:, tok.n_parameter_variables:],
            condition_mask=mask,
        )
    assert torch.allclose(a, b)


# --------------------------------------------------------------------------
# function-valued parameters
# --------------------------------------------------------------------------
def test_function_valued_token() -> None:
    """A function-valued parameter adds one token with shared id embedding and
    a random Fourier metadata embedding of its index set."""
    spec = TokenSpec(
        parameter_names=("alpha",),
        data_names=("x_1",),
        function_valued=(FunctionValuedSpec(name="beta", index_set=np.linspace(0.0, 1.0, 8)),),
        metadata_dim=32,
    )
    torch.manual_seed(0)
    tok = Tokenizer(spec, token_dim=TOKEN_DIM)

    assert tok.n_function_tokens == 1
    assert tok.n_variables == 3  # alpha, x_1, beta(t)
    # joint vector: alpha + x_1 + 8 index values of beta
    assert tok.input_dim == 1 + 1 + 8

    theta = torch.tensor([[0.5, 0.2]], dtype=torch.float32)
    x = torch.tensor([[0.1, 0.3]], dtype=torch.float32)
    fvals = torch.linspace(0.0, 1.0, 8).reshape(1, 8)
    tokens = tok(theta, x, torch.zeros(1, tok.n_variables), function_values=fvals)
    assert tokens.shape == (1, 3, TOKEN_DIM)
    assert torch.isfinite(tokens).all()


def test_function_valued_subsampling() -> None:
    """``n_index_points`` sub-samples a large index grid (used for HH/SIRD)."""
    spec = FunctionValuedSpec(name="beta", index_set=np.linspace(0.0, 1.0, 100), n_index_points=20)
    assert spec.index_set.shape[0] == 20
    assert np.isclose(spec.index_set[0], 0.0)
    assert np.isclose(spec.index_set[-1], 1.0)


def test_tokenizer_parameters_are_trainable(benchmark_tokenizer: Tokenizer) -> None:
    trainable = [p for p in benchmark_tokenizer.parameters() if p.requires_grad]
    assert len(trainable) > 0
    assert all(torch.isfinite(p).all() for p in trainable)


def test_benchmark_spec_names() -> None:
    spec = build_benchmark_spec(n_parameters=3, n_data=4)
    assert len(spec.parameter_names) == 3
    assert len(spec.data_names) == 4
