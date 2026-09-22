import numpy as np

from wwmf.evaluation.metrics import (
    binary_f1_scores,
    edit_success_rate,
    em_drop_ratio,
    exact_match,
    forget_frequency,
    normalize_answer,
    precision_recall_f1,
)


def test_normalize_answer_follows_squad20():
    # lower-casing, punctuation removal, article removal, whitespace collapse
    assert normalize_answer("The Boston Celtics!") == "boston celtics"
    assert normalize_answer("a  an THE") == ""
    assert normalize_answer("Hello,   World\n") == "hello world"


def test_exact_match_uses_normalisation():
    assert exact_match(["The answer."], ["answer"]) == 1.0
    assert exact_match(["yes"], ["no"]) == 0.0
    assert exact_match(["A", "b", "c"], ["a", "B", "d"]) == 2 / 3
    assert edit_success_rate(["True"], ["true"]) == 1.0


def test_em_drop_ratio_matches_definition():
    assert abs(em_drop_ratio(0.45, 0.50) - (-0.10)) < 1e-12
    assert em_drop_ratio(0.5, 0.5) == 0.0


def test_binary_f1_scores():
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 1, 0]
    precision, recall, f1 = precision_recall_f1(y_true, y_pred)
    assert abs(precision - 50.0) < 1e-9
    assert abs(recall - 50.0) < 1e-9
    assert abs(f1 - 50.0) < 1e-9
    scores = binary_f1_scores(y_true, y_pred)
    assert set(scores) == {"precision", "recall", "f1"}
    # no positive predictions -> zero precision/recall/f1
    assert binary_f1_scores([0, 0], [0, 0])["f1"] == 0.0


def test_forget_frequency_counts_online_examples():
    labels = [[1, 0, 0], [1, 1, 0], [0, 0, 0]]
    freq = forget_frequency(labels)
    assert np.allclose(freq, [2 / 3, 1 / 3, 0.0])
