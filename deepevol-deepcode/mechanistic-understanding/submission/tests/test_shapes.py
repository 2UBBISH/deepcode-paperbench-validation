"""Shape / architecture sanity tests for the DPO-toxicity reproduction.

These tests verify the tensor conventions that every downstream component of the
reproduction relies on (Section 2 / Section 3.1 of the paper):

* GPT2-medium has ``L = 24`` transformer layers, ``d_model = 1024`` and
  ``d_mlp = 4096`` (the paper's ``l``, ``d``, ``d_mlp``).
* MLP **value** vectors ``MLP.v_i^l`` live in the residual stream space
  ``R^{d_model}``; there are ``d_mlp`` of them per layer, hence
  ``d_mlp * L`` value/key vectors in total.
* The residual stream captured *after attention, before the MLP* (the paper's
  ``x^{l-mid}``) has shape ``[L, batch, seq, d_model]``.
* Vocabulary projections ``r = E v`` have shape ``[..., vocab]``.
* Key vectors are views into the MLP key matrix, so
  ``scale_key_vector`` (used by the Section 6 un-alignment experiment) modifies
  the parameters in place and can be restored.

The tests are designed to run without network access: if GPT2-medium is not
available/cached they fall back to a tiny randomly initialised GPT2 built from
``GPT2Config`` (identical architecture conventions, tiny dimensions).  Set
``REPRO_MODEL=openai-community/gpt2-medium`` (default) to test the real model.

Run with either ``python -m pytest tests/test_shapes.py -q`` or
``python -m unittest tests.test_shapes -v``.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src import model_utils  # noqa: E402
from src.model_utils import (  # noqa: E402
    GPT2_MEDIUM,
    all_key_vectors,
    all_value_vectors,
    capture_residual_streams,
    forward_with_residuals,
    get_embedding,
    get_key_vector,
    get_mlp_matrices,
    get_unembedding,
    get_value_vector,
    model_info,
    project_to_vocab,
    scale_key_vector,
    transformer_layers,
    unembed_hidden_state,
)

# Paper constants for GPT2-medium (Section 2).
GPT2_MEDIUM_L = 24
GPT2_MEDIUM_D = 1024
GPT2_MEDIUM_D_MLP = 4096

# Tiny fallback architecture (same conventions, cheap on CPU).
TINY_LAYERS = 4
TINY_D = 64
TINY_D_MLP = 256
TINY_VOCAB = 512
TINY_SEQ = 16


def _build_tiny_gpt2():
    """Create a tiny randomly-initialised GPT2 (no network access)."""
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=TINY_VOCAB,
        n_positions=64,
        n_ctx=64,
        n_embd=TINY_D,
        n_layer=TINY_LAYERS,
        n_head=4,
        n_inner=TINY_D_MLP,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        bos_token_id=0,
        eos_token_id=0,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    return model, config


def _load_real_or_tiny():
    """Load GPT2-medium when available, else the tiny fallback.

    Returns ``(model, tokenizer_or_None, is_real, info)``.
    """
    model_name = os.environ.get("REPRO_MODEL", GPT2_MEDIUM)
    force_tiny = os.environ.get("REPRO_TINY", "").lower() in {"1", "true", "yes"}
    if not force_tiny:
        try:
            model, tokenizer = model_utils.load_model(model_name, device="cpu")
            return model, tokenizer, True, model_info(model, name=model_name)
        except Exception as exc:  # pragma: no cover - offline fallback
            print(f"[test_shapes] falling back to tiny GPT2 ({type(exc).__name__}: {exc})")
    model, _ = _build_tiny_gpt2()
    return model, None, False, model_info(model, name="tiny-gpt2")


class GPT2ArchitectureTests(unittest.TestCase):
    """GPT2-medium architecture constants as used throughout the paper."""

    @classmethod
    def setUpClass(cls):
        cls.model, cls.tokenizer, cls.is_real, cls.info = _load_real_or_tiny()

    def test_model_info_matches_gpt2_medium(self):
        info = self.info
        if self.is_real:
            self.assertEqual(info.n_layers, GPT2_MEDIUM_L)
            self.assertEqual(info.d_model, GPT2_MEDIUM_D)
            self.assertEqual(info.d_mlp, GPT2_MEDIUM_D_MLP)
            self.assertEqual(info.vocab_size, 50257)
            self.assertFalse(info.is_glu)
        else:
            self.assertEqual(info.n_layers, TINY_LAYERS)
            self.assertEqual(info.d_model, TINY_D)
            self.assertEqual(info.d_mlp, TINY_D_MLP)

    def test_transformer_layers_count(self):
        layers = transformer_layers(self.model)
        self.assertEqual(len(layers), self.info.n_layers)

    def test_mlp_matrices_have_d_mlp_vectors_of_dim_d_model(self):
        W_K, W_V = get_mlp_matrices(self.model, 0)
        self.assertEqual(tuple(W_K.shape), (self.info.d_mlp, self.info.d_model))
        self.assertEqual(tuple(W_V.shape), (self.info.d_mlp, self.info.d_model))

    def test_value_and_key_vector_shapes(self):
        layer = self.info.n_layers - 1
        idx = self.info.d_mlp - 1
        v = get_value_vector(self.model, layer, idx)
        k = get_key_vector(self.model, layer, idx)
        self.assertEqual(tuple(v.shape), (self.info.d_model,))
        self.assertEqual(tuple(k.shape), (self.info.d_model,))

    def test_all_vectors_stacks_match_paper_count(self):
        values, v_indices = all_value_vectors(self.model)
        keys, k_indices = all_key_vectors(self.model)
        expected = self.info.n_layers * self.info.d_mlp
        self.assertEqual(values.shape, (expected, self.info.d_model))
        self.assertEqual(keys.shape, (expected, self.info.d_model))
        self.assertEqual(len(v_indices), expected)
        self.assertEqual(len(k_indices), expected)
        # Index tuples are (layer, idx) in range.
        l0, i0 = v_indices[0]
        self.assertEqual(l0, 0)
        self.assertEqual(i0, 0)


class ResidualStreamShapeTests(unittest.TestCase):
    """``x^{l-mid}`` (post-attention, pre-MLP) capture conventions."""

    @classmethod
    def setUpClass(cls):
        cls.model, cls.tokenizer, cls.is_real, cls.info = _load_real_or_tiny()

    def _input_ids(self, batch=2, seq=8):
        torch.manual_seed(0)
        return torch.randint(0, min(self.info.vocab_size, 1000), (batch, seq))

    def test_capture_shapes(self):
        input_ids = self._input_ids()
        with capture_residual_streams(self.model, capture_mlp_act=True) as cap:
            with torch.no_grad():
                self.model(input_ids)
        mid = cap.mid_tensor()
        self.assertEqual(mid.dim(), 4)
        self.assertEqual(mid.shape[0], self.info.n_layers)
        self.assertEqual(mid.shape[1], 2)
        self.assertEqual(mid.shape[2], 8)
        self.assertEqual(mid.shape[3], self.info.d_model)
        # Per-layer accessors agree with the stacked tensor.
        self.assertEqual(tuple(cap.get_mid(0).shape), (2, 8, self.info.d_model))
        self.assertEqual(
            tuple(cap.get_block_out(self.info.n_layers - 1).shape),
            (2, 8, self.info.d_model),
        )
        self.assertIsNotNone(cap.mlp_act)
        self.assertEqual(len(cap.mlp_act), self.info.n_layers)

    def test_forward_with_residuals(self):
        input_ids = self._input_ids()
        outputs, mid, block_out, _ = forward_with_residuals(self.model, input_ids)
        self.assertTrue(hasattr(outputs, "logits"))
        self.assertEqual(outputs.logits.shape, (2, 8, self.info.vocab_size))
        self.assertEqual(mid.shape, (self.info.n_layers, 2, 8, self.info.d_model))
        self.assertEqual(block_out.shape, (self.info.n_layers, 2, 8, self.info.d_model))


class VocabularyProjectionTests(unittest.TestCase):
    """Section 3.2 / Table 1 vocabulary-space projection conventions."""

    @classmethod
    def setUpClass(cls):
        cls.model, cls.tokenizer, cls.is_real, cls.info = _load_real_or_tiny()

    def test_embedding_and_unembedding_shapes(self):
        E = get_embedding(self.model)
        U = get_unembedding(self.model)
        self.assertEqual(tuple(E.shape), (self.info.vocab_size, self.info.d_model))
        self.assertEqual(tuple(U.shape), (self.info.vocab_size, self.info.d_model))
        # GPT2 ties the unembedding to the input embedding.
        self.assertEqual(E.shape, U.shape)

    def test_project_to_vocab_matches_matmul(self):
        v = get_value_vector(self.model, 0, 0).detach()
        r = project_to_vocab(self.model, v)
        self.assertEqual(tuple(r.shape), (self.info.vocab_size,))
        E = get_embedding(self.model).detach()
        manual = E @ v
        self.assertTrue(torch.allclose(r.detach(), manual, atol=1e-5))

    def test_project_batch_of_vectors(self):
        vectors = torch.stack([get_value_vector(self.model, 0, i).detach() for i in range(3)])
        r = project_to_vocab(self.model, vectors)
        self.assertEqual(tuple(r.shape), (3, self.info.vocab_size))

    def test_unembed_hidden_state(self):
        hidden = torch.randn(2, 5, self.info.d_model)
        logits = unembed_hidden_state(self.model, hidden)
        self.assertEqual(tuple(logits.shape), (2, 5, self.info.vocab_size))


class KeyVectorScalingTests(unittest.TestCase):
    """Section 6 un-alignment: key vectors must be writable views."""

    @classmethod
    def setUpClass(cls):
        cls.model, cls.tokenizer, cls.is_real, cls.info = _load_real_or_tiny()

    def test_scale_key_vector_is_in_place_and_reversible(self):
        layer, idx = 0, 1
        W_K, _ = get_mlp_matrices(self.model, layer)
        original = W_K[idx].detach().clone()
        scaled = scale_key_vector(self.model, layer, idx, scale=10.0)
        W_K_after, _ = get_mlp_matrices(self.model, layer)
        self.assertTrue(torch.allclose(W_K_after[idx].detach(), original * 10.0, atol=1e-4))
        self.assertTrue(torch.allclose(scaled.detach(), original * 10.0, atol=1e-4))
        # Restore.
        scale_key_vector(self.model, layer, idx, scale=0.1)
        W_K_restored, _ = get_mlp_matrices(self.model, layer)
        self.assertTrue(torch.allclose(W_K_restored[idx].detach(), original, atol=1e-4))

    def test_scaling_does_not_change_value_vectors(self):
        layer, idx = 0, 2
        before = get_value_vector(self.model, layer, idx).detach().clone()
        scale_key_vector(self.model, layer, idx, scale=10.0)
        try:
            after = get_value_vector(self.model, layer, idx).detach()
            self.assertTrue(torch.allclose(before, after, atol=1e-6))
        finally:
            scale_key_vector(self.model, layer, idx, scale=0.1)


class TinyForwardShapeTests(unittest.TestCase):
    """Forward pass and activation shapes for the tiny fallback model."""

    def setUp(self):
        self.model, _ = _build_tiny_gpt2()
        self.info = model_info(self.model, name="tiny-gpt2")

    def test_mlp_activation_sigma_shape(self):
        from src.model_utils import activation_sigma

        x = torch.randn(2, 3, TINY_D_MLP)
        y = activation_sigma(self.model, x)
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        # GeLU is not the identity.
        self.assertFalse(torch.allclose(x, y, atol=1e-3))

    def test_glu_helpers_marked_out_of_scope(self):
        """GLU/Llama2 helpers are documented stubs (out of reproduction scope)."""
        self.assertTrue(hasattr(model_utils, "glu_value_vectors"))
        self.assertFalse(model_utils.is_glu_model("openai-community/gpt2-medium"))

    def test_numpy_pipeline_shapes(self):
        """Numeric stacking used by probe / toxic-vector extraction."""
        values, indices = all_value_vectors(self.model)
        self.assertEqual(values.shape[0], len(indices))
        arr = np.asarray(values, dtype=np.float32)
        self.assertEqual(arr.shape, (TINY_LAYERS * TINY_D_MLP, TINY_D))
        # Cosine similarities against a direction -> one score per value vector.
        direction = np.ones(TINY_D, dtype=np.float32)
        cosines = arr @ direction / (np.linalg.norm(arr, axis=1) * np.linalg.norm(direction) + 1e-12)
        self.assertEqual(cosines.shape[0], TINY_LAYERS * TINY_D_MLP)


if __name__ == "__main__":
    unittest.main(verbosity=2)
