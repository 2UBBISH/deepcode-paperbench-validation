"""Integration tests that exercise the paper's experiment code paths."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lca_on_the_line.analysis import (  # noqa: E402
    compute_ranking_table,
    compute_table2,
    compute_table3,
    format_ranking_table,
    load_metrics,
    metric_table,
)
from lca_on_the_line.evaluate import METRIC_FIELDS  # noqa: E402
from lca_on_the_line.latent import (  # noqa: E402
    class_features_from_logits_and_features,
    latent_hierarchy_correlation,
)
from lca_on_the_line.soft_labels import (  # noqa: E402
    ProbeConfig,
    run_soft_label_experiment,
)


# --------------------------------------------------------------------------- #
# Section 4.3.2: soft-label linear probing
# --------------------------------------------------------------------------- #
def test_soft_label_experiment_improves_or_matches_baseline():
    rng = np.random.RandomState(0)
    n_classes, dim = 8, 16
    centroids = rng.randn(n_classes, dim)
    # classes 0/1 and 2/3 are close pairs in the taxonomy
    centroids[1] = centroids[0] + 0.4 * rng.randn(dim)
    centroids[3] = centroids[2] + 0.4 * rng.randn(dim)
    train_y = rng.randint(0, n_classes, size=600)
    train_x = centroids[train_y] + 0.7 * rng.randn(600, dim)
    eval_y = rng.randint(0, n_classes, size=200)
    eval_x = centroids[eval_y] + 0.7 * rng.randn(200, dim)

    lca = np.full((n_classes, n_classes), 3.0)
    np.fill_diagonal(lca, 0.0)
    lca[0, 1] = lca[1, 0] = 1.0
    lca[2, 3] = lca[3, 2] = 1.0

    results = run_soft_label_experiment(
        {
            "train": (torch.tensor(train_x, dtype=torch.float32),
                      torch.tensor(train_y)),
            "imagenet": (torch.tensor(eval_x, dtype=torch.float32),
                         torch.tensor(eval_y)),
        },
        lca,
        config=ProbeConfig(epochs=4, batch_size=128, lr=5e-3, lambda_weight=0.03,
                           temperature=25.0),
    )
    assert "imagenet/baseline" in results
    assert "imagenet/soft" in results
    # interpolation with alpha=1 recovers the plain CE probe
    assert results["imagenet/interp@1.00"] == pytest.approx(
        results["imagenet/baseline"], abs=1e-6
    )
    assert max(results.values()) > 0.3


# --------------------------------------------------------------------------- #
# Section 4.3.1: latent hierarchies
# --------------------------------------------------------------------------- #
def test_class_features_are_class_means():
    feats = np.array([[1.0, 1.0], [3.0, 3.0], [2.0, 0.0], [4.0, 0.0]])
    targets = [0, 0, 1, 1]
    means = class_features_from_logits_and_features(feats, targets, n_classes=2)
    assert np.allclose(means[0], [2.0, 2.0])
    assert np.allclose(means[1], [3.0, 0.0])


def test_latent_hierarchy_correlation_is_perfect_for_correlated_data():
    rng = np.random.RandomState(0)
    n_classes, n_models, n_samples = 30, 12, 500
    targets = rng.randint(0, n_classes, size=n_samples)
    # a cyclic taxonomy: severity grows with the class offset
    matrix = np.abs(
        np.arange(n_classes)[:, None] - np.arange(n_classes)[None, :]
    ).astype(float)
    matrix = np.minimum(matrix, n_classes - matrix)

    # models with decreasing quality make more *and more severe* mistakes
    id_preds, ood_preds = [], []
    for m in range(n_models):
        noise = 0.05 + 0.04 * m
        severity = 1 + m // 2
        id_pred = targets.copy()
        flip = rng.rand(n_samples) < noise
        offsets = rng.randint(1, severity + 1, size=int(flip.sum()))
        id_pred[flip] = (targets[flip] + offsets) % n_classes
        ood_pred = targets.copy()
        flip = rng.rand(n_samples) < (0.1 + 0.05 * m)
        offsets = rng.randint(1, severity + 1, size=int(flip.sum()))
        ood_pred[flip] = (targets[flip] + offsets) % n_classes
        id_preds.append(id_pred)
        ood_preds.append(ood_pred)

    result = latent_hierarchy_correlation(
        id_preds, ood_preds, targets, [matrix, matrix * 2.0]
    )
    assert result["mean"] > 0.6
    assert len(result["per_matrix_pea"]) == 2


# --------------------------------------------------------------------------- #
# Section 4.2 + Table 3 end-to-end on cached logits
# --------------------------------------------------------------------------- #
def _write_synthetic_results(out_dir, n_models=8, n_classes=10, n_samples=600):
    rng = np.random.RandomState(0)
    os.makedirs(os.path.join(out_dir, "logits"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "targets"), exist_ok=True)
    targets = rng.randint(0, n_classes, size=n_samples)
    np.save(os.path.join(out_dir, "targets", "imagenet.npy"), targets)
    np.save(os.path.join(out_dir, "targets", "imagenet_v2.npy"), targets)

    rows = []
    for m in range(n_models):
        logits_id = rng.randn(n_samples, n_classes) * 0.8
        logits_id[np.arange(n_samples), targets] += 2.5 - 0.15 * m
        logits_ood = rng.randn(n_samples, n_classes) * 0.8
        logits_ood[np.arange(n_samples), targets] += 2.2 - 0.15 * m
        np.save(os.path.join(out_dir, "logits", "model%d__imagenet.npy" % m), logits_id)
        np.save(os.path.join(out_dir, "logits", "model%d__imagenet_v2.npy" % m), logits_ood)
        for dataset, logits in (("imagenet", logits_id), ("imagenet_v2", logits_ood)):
            preds = logits.argmax(axis=1)
            top1 = float((preds == targets).mean())
            rows.append({
                "model": "model%d" % m, "family": "VM" if m % 2 else "VLM",
                "source": "synthetic", "dataset": dataset,
                "top1": top1,
                # Top-5 must vary across models or the rank correlations are nan
                "top5": min(1.0, top1 + 0.15 + 0.01 * m),
                "lca": float(rng.rand()), "elca": 0.0,
                "n_samples": n_samples, "n_mistakes": int((preds != targets).sum()),
                "seconds": 0.0,
            })
    import csv

    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out_dir


def test_table2_and_table3_on_synthetic_logits(tmp_path):
    out_dir = _write_synthetic_results(str(tmp_path))
    df = load_metrics(out_dir)
    table2 = compute_table2(df, ood_datasets=["imagenet_v2"])
    assert set(table2) == {"Top1->TOP1", "Top1->TOP5", "LCA->TOP1", "LCA->TOP5"}
    assert "ImgN-v2" in table2["LCA->TOP1"]

    table3 = compute_table3(df, out_dir, ood_datasets=["imagenet_v2"])
    assert "ImgN-v2" in table3.columns
    assert "(Ours) ID LCA" in table3.index
    assert "Aline-D (Baek et al., 2022)" in table3.index
    for value in table3["ImgN-v2"].values:
        assert np.isfinite(value)


def test_metric_table_shapes(tmp_path):
    out_dir = _write_synthetic_results(str(tmp_path))
    df = load_metrics(out_dir)
    table = metric_table(df, "top1")
    assert table.shape[0] == 8
    assert set(table.columns) == {"imagenet", "imagenet_v2"}


def test_ranking_table(tmp_path):
    out_dir = _write_synthetic_results(str(tmp_path))
    df = load_metrics(out_dir)
    ranking = compute_ranking_table(df, ood_datasets=["imagenet_v2"])
    assert set(ranking) == {"Top1->TOP1", "Top1->TOP5", "LCA->TOP1", "LCA->TOP5"}
    for values in ranking.values():
        for metrics in values.values():
            assert -1.0 <= metrics["KEN"] <= 1.0
            assert -1.0 <= metrics["SPE"] <= 1.0
    assert "ImgN-v2" in format_ranking_table(ranking)


# --------------------------------------------------------------------------- #
# Figures 7 and 8
# --------------------------------------------------------------------------- #
def test_figure7_and_figure8_render(tmp_path):
    from lca_on_the_line.analysis import figure7, figure8
    from lca_on_the_line.hierarchy import load_wordnet_hierarchy

    hierarchy = load_wordnet_hierarchy()
    matrix = hierarchy.lca_distance_matrix("information")
    rng = np.random.RandomState(0)
    out7 = str(tmp_path / "figure7.png")
    figure7(
        {"WordNet": matrix, "latent": rng.rand(200, 200) * matrix.max()},
        out7,
    )
    assert os.path.exists(out7)

    out8 = str(tmp_path / "figure8.png")
    figure8(rng.rand(25) * 8, rng.rand(25), out8)
    assert os.path.exists(out8)


# --------------------------------------------------------------------------- #
# Appendix E.4 / Table 10 (described in the main text of Section 4.3.2)
# --------------------------------------------------------------------------- #
def test_soft_label_quality_correlation(tmp_path, monkeypatch):
    import torch

    from lca_on_the_line import experiments_latent, experiments_soft_labels as esl

    out_dir = _write_synthetic_results(str(tmp_path), n_models=4, n_classes=10,
                                       n_samples=200)
    n_classes, dim = 10, 12
    rng = np.random.RandomState(0)

    features = {}
    for split in ("train", "imagenet", "imagenet_v2"):
        y = rng.randint(0, n_classes, size=300 if split == "train" else 120)
        x = rng.randn(len(y), dim) + y[:, None] * 0.5
        features[split] = (torch.tensor(x, dtype=torch.float32), torch.tensor(y))
    monkeypatch.setattr(esl, "extract_backbone_features",
                        lambda *a, **k: features)

    def fake_class_features(name, data_root, per_class=20, device="cpu", **kw):
        return rng.randn(n_classes, dim)

    monkeypatch.setattr(
        experiments_latent, "class_features_for_model", fake_class_features
    )

    study = esl.soft_label_quality_correlation(
        "resnet18",
        data_root="/tmp",
        results_dir=out_dir,
        source_models=["model0", "model1", "model2", "model3"],
        limit_eval=120,
        config=esl.ProbeConfig(epochs=2, batch_size=64),
    )
    assert "PEA" in study
    assert set(study["PEA"]) == {"soft_labels", "ce_baseline"}
    assert len(study["source_lca"]) == 4
    assert "imagenet_v2" in study["PEA"]["soft_labels"]
