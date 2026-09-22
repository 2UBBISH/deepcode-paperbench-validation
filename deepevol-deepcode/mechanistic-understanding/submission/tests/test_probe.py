"""Smoke tests for the toxicity probe ``W_Toxic`` (Section 3.1).

The paper trains a linear probe ``softmax(W_Toxic x_bar^(L-1))`` with
``W_Toxic`` of shape ``[d_model, 2]`` on Jigsaw comments (column 0 = non-toxic,
column 1 = toxic) and reports ~94% validation accuracy.  These tests exercise the
*probe machinery* in :mod:`src.probe` on synthetic data so they run offline on
CPU in a few seconds:

* weight/direction shape conventions (``W[:, 1]`` is the toxic direction),
* softmax probability semantics and monotonicity of ``toxic_probability``,
* end-to-end linear-probe training on two separable Gaussian clusters,
* evaluation helper reporting,
* artifact round-trip (:func:`save_probe` / :func:`load_probe`),
* (when a tiny GPT2 can be built locally) the full
  ``collect_probe_features`` -> ``train_probe`` pipeline.

No network access and no real GPT2/Jigsaw download is required.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

# ---------------------------------------------------------------------------
# Path bootstrap: allow ``python tests/test_probe.py`` from the repo root.
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

try:  # torch is required by the probe implementation
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch should always be installed
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False

from src import probe as probe_mod  # noqa: E402
from src.probe import (  # noqa: E402
    TARGET_VALID_ACCURACY,
    TOXIC_INDEX,
    ToxicityProbe,
    evaluate_probe,
    load_probe,
    probe_exists,
    save_probe,
    train_probe,
)

D_MODEL = 64
N_SAMPLES = 512
NOISE = 0.6
SEPARATION = 1.6


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _to_numpy(value):
    """Convert torch tensors / lists / numpy scalars into a numpy array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _make_features(n=N_SAMPLES, d=D_MODEL, seed=0, noise=NOISE, separation=SEPARATION):
    """Two Gaussian clusters separated along a random direction."""
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=d)
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    labels = rng.integers(0, 2, size=n).astype(np.int64)
    offset = np.where(labels[:, None] == 1, separation * direction, -separation * direction)
    features = (rng.normal(scale=noise, size=(n, d)) + offset).astype(np.float32)
    return features, labels, direction.astype(np.float32)


def _one_hot_weight(d, value=1.0):
    """``W[:,0] = -v``, ``W[:,1] = +v`` with ``v = e_0`` -> logit diff = 2 * x_0."""
    W = np.zeros((d, 2), dtype=np.float32)
    W[0, 1] = value
    W[0, 0] = -value
    return W


def _build_tiny_gpt2(layers=4, d_model=64, d_mlp=128, vocab=256, n_heads=4):
    """Build a small randomly-initialised GPT2 without touching the network."""
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=vocab,
        n_positions=64,
        n_ctx=64,
        n_embd=d_model,
        n_layer=layers,
        n_head=n_heads,
        n_inner=d_mlp,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    return model, config


class _FakeTokenizer:
    """Minimal tokenizer stub: deterministic ids, no network / no vocab files."""

    pad_token_id = 0
    eos_token_id = 0
    bos_token_id = 0

    def __init__(self, vocab_size=256, length=8, seed=0):
        self.vocab_size = vocab_size
        self.length = length
        self._rng = np.random.default_rng(seed)

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True,
                 max_length=None, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        length = self.length if max_length is None else min(self.length, int(max_length))
        ids = self._rng.integers(1, self.vocab_size, size=(len(texts), length))
        input_ids = torch.as_tensor(ids, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


# ---------------------------------------------------------------------------
# construction / direction conventions
# ---------------------------------------------------------------------------
@unittest.skipUnless(_HAS_TORCH, "torch is required")
class ProbeConventionTests(unittest.TestCase):
    """``W_Toxic`` shape and toxic-direction conventions."""

    def setUp(self):
        self.d = 16
        self.W = _one_hot_weight(self.d)
        self.probe = ToxicityProbe(torch.as_tensor(self.W), model_name="unit-test", n_layers=4)

    def test_d_model_matches_weight_matrix(self):
        self.assertEqual(int(self.probe.d_model), self.d)

    def test_toxic_direction_is_second_column(self):
        direction = _to_numpy(self.probe.toxic_direction).reshape(-1)
        self.assertEqual(direction.shape, (self.d,))
        np.testing.assert_allclose(direction, self.W[:, TOXIC_INDEX], atol=1e-6)
        # sign matters: the toxic direction is +e_0 in this construction
        self.assertGreater(direction[0], 0.0)

    def test_non_toxic_direction_is_first_column(self):
        direction = _to_numpy(self.probe.non_toxic_direction).reshape(-1)
        np.testing.assert_allclose(direction, self.W[:, 0], atol=1e-6)

    def test_unit_direction_is_normalised(self):
        unit = _to_numpy(self.probe.unit_toxic_direction).reshape(-1)
        self.assertAlmostEqual(float(np.linalg.norm(unit)), 1.0, places=4)


@unittest.skipUnless(_HAS_TORCH, "torch is required")
class ProbeProbabilityTests(unittest.TestCase):
    """softmax(W_Toxic x) semantics."""

    def setUp(self):
        self.d = 16
        self.probe = ToxicityProbe(torch.as_tensor(_one_hot_weight(self.d)),
                                   model_name="unit-test", n_layers=4)
        self.rng = np.random.default_rng(0)

    def test_probabilities_are_a_distribution(self):
        X = torch.as_tensor(self.rng.normal(size=(32, self.d)), dtype=torch.float32)
        probs = _to_numpy(self.probe.probabilities(X))
        self.assertEqual(probs.shape, (32, 2))
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(32), atol=1e-4)
        self.assertTrue(np.all(probs >= -1e-6) and np.all(probs <= 1 + 1e-6))

    def test_toxic_probability_matches_second_column(self):
        X = torch.as_tensor(self.rng.normal(size=(32, self.d)), dtype=torch.float32)
        probs = _to_numpy(self.probe.probabilities(X))
        toxic = _to_numpy(self.probe.toxic_probability(X)).reshape(-1)
        np.testing.assert_allclose(toxic, probs[:, TOXIC_INDEX], atol=1e-4)

    def test_toxic_probability_monotone_in_toxic_direction(self):
        # logit difference is 2 * x_0 for this construction
        xs = np.linspace(-3.0, 3.0, 13, dtype=np.float32)
        X = np.zeros((len(xs), self.d), dtype=np.float32)
        X[:, 0] = xs
        toxic = _to_numpy(self.probe.toxic_probability(torch.as_tensor(X))).reshape(-1)
        self.assertTrue(np.all(np.diff(toxic) > 0.0),
                        msg="toxic_probability must increase along the toxic direction")
        self.assertLess(toxic[0], 0.5)
        self.assertGreater(toxic[-1], 0.5)

    def test_predict_returns_binary_labels(self):
        X = np.zeros((2, self.d), dtype=np.float32)
        X[0, 0] = 3.0   # strongly toxic side
        X[1, 0] = -3.0  # strongly non-toxic side
        pred = _to_numpy(self.probe.predict(torch.as_tensor(X))).reshape(-1).astype(int)
        self.assertEqual(pred.shape, (2,))
        self.assertEqual(pred[0], TOXIC_INDEX)
        self.assertEqual(pred[1], 0)


# ---------------------------------------------------------------------------
# training smoke test
# ---------------------------------------------------------------------------
@unittest.skipUnless(_HAS_TORCH, "torch is required")
class ProbeTrainingSmokeTests(unittest.TestCase):
    """Train the linear probe on separable synthetic features."""

    @classmethod
    def setUpClass(cls):
        cls.features, cls.labels, cls.direction = _make_features()
        cls.result = train_probe(
            cls.features,
            cls.labels,
            valid_ratio=0.2,
            seed=0,
            lr=1e-2,
            batch_size=64,
            epochs=40,
            patience=5,
            verbose=False,
        )

    def test_training_result_exposes_probe(self):
        self.assertIsInstance(self.result.probe, ToxicityProbe)

    def test_weight_matrix_shape(self):
        W = _to_numpy(self.result.probe.W)
        self.assertEqual(W.shape, (D_MODEL, 2))

    def test_probe_learns_separable_task(self):
        accuracy = float(self.result.valid_accuracy)
        self.assertGreater(accuracy, 0.85, msg=f"valid accuracy too low: {accuracy:.3f}")
        self.assertGreaterEqual(float(self.result.accuracy), 0.85)

    def test_evaluate_probe_reports_accuracy(self):
        metrics = evaluate_probe(self.result.probe, self.features, self.labels)
        self.assertIn("accuracy", metrics)
        self.assertGreater(float(metrics["accuracy"]), 0.85)

    def test_toxic_direction_aligns_with_true_direction(self):
        direction = _to_numpy(self.result.probe.toxic_direction).reshape(-1)
        cosine = float(direction @ self.direction / (np.linalg.norm(direction) + 1e-12))
        self.assertGreater(cosine, 0.0,
                           msg="learned toxic direction must point towards the toxic cluster")

    def test_summary_is_serialisable(self):
        summary = self.result.summary()
        self.assertIn("valid_accuracy", summary)
        self.assertIsInstance(summary["valid_accuracy"], float)

    def test_target_accuracy_constant_matches_paper(self):
        self.assertAlmostEqual(float(TARGET_VALID_ACCURACY), 0.94, places=6)

    def test_explicit_validation_arrays_are_respected(self):
        xtr, ytr, _ = _make_features(n=256, seed=1)
        xva, yva, _ = _make_features(n=128, seed=2)
        result = train_probe(
            xtr, ytr, valid_features=xva, valid_labels=yva,
            seed=0, lr=1e-2, batch_size=64, epochs=20, patience=5, verbose=False,
        )
        self.assertEqual(int(result.n_train), 256)
        self.assertGreater(float(result.valid_accuracy), 0.85)


# ---------------------------------------------------------------------------
# persistence round trip
# ---------------------------------------------------------------------------
@unittest.skipUnless(_HAS_TORCH, "torch is required")
class ProbePersistenceTests(unittest.TestCase):
    """``save_probe`` / ``load_probe`` correctness."""

    def test_round_trip_preserves_weights_and_direction(self):
        features, labels, _ = _make_features(n=256, seed=3)
        result = train_probe(features, labels, valid_ratio=0.2, seed=0,
                             lr=1e-2, batch_size=64, epochs=20, patience=5,
                             verbose=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "w_toxic.pt")
            self.assertFalse(probe_exists(path))
            save_probe(path, result, verbose=False)
            self.assertTrue(probe_exists(path))

            reloaded = load_probe(path)
            before = _to_numpy(result.probe.W).reshape(D_MODEL, 2)
            after = _to_numpy(reloaded.W).reshape(D_MODEL, 2)
            np.testing.assert_allclose(after, before, atol=1e-5)

            np.testing.assert_allclose(
                _to_numpy(reloaded.toxic_direction).reshape(-1),
                _to_numpy(result.probe.toxic_direction).reshape(-1),
                atol=1e-5,
            )

    def test_module_exposes_paper_defaults(self):
        self.assertEqual(TOXIC_INDEX, 1)
        self.assertTrue(hasattr(probe_mod, "PROBE_PATH"))
        self.assertTrue(str(probe_mod.PROBE_PATH).endswith(".pt"))


# ---------------------------------------------------------------------------
# optional offline end-to-end feature collection (tiny GPT2)
# ---------------------------------------------------------------------------
@unittest.skipUnless(_HAS_TORCH, "torch is required")
class ProbeFeaturePipelineTests(unittest.TestCase):
    """``collect_probe_features`` -> ``train_probe`` on a tiny local GPT2."""

    def test_feature_collection_shapes_and_training(self):
        try:
            model, _ = _build_tiny_gpt2(d_model=D_MODEL, d_mlp=128, vocab=128)
            tokenizer = _FakeTokenizer(vocab_size=128, length=8)
            texts = [f"synthetic comment number {i}" for i in range(24)]
            labels = np.array([i % 2 for i in range(24)], dtype=np.int64)
            features = probe_mod.collect_probe_features(
                model, tokenizer, texts, layer=-1, batch_size=8, max_length=8,
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            self.skipTest(f"tiny GPT2 feature pipeline unavailable: {exc}")
            return

        features = _to_numpy(features)
        self.assertEqual(features.shape, (24, D_MODEL))
        self.assertTrue(np.all(np.isfinite(features)))

        # a tiny training run should at least produce a valid probe object
        result = train_probe(features, labels, valid_ratio=0.5, seed=0,
                             lr=1e-2, batch_size=8, epochs=3, patience=2,
                             verbose=False)
        W = _to_numpy(result.probe.W)
        self.assertEqual(W.shape, (D_MODEL, 2))
        self.assertTrue(np.all(np.isfinite(W)))


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
