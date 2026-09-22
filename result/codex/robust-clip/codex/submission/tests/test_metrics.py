"""Tests of the CIDEr / VQA / POPE metrics."""

from robust_clip.eval.cider import Cider, tokenize
from robust_clip.eval.lvlm_eval import parse_sqa_answer, parse_yes_no, pope_f1
from robust_clip.eval.vqa import most_frequent_answers, normalize_answer, vqa_accuracy


def _corpus():
    captions = [
        ["a cat sits on the mat", "the cat is on the mat", "a cat on a mat"],
        ["a dog runs in the park", "a dog running through a park", "dog in the park"],
        ["a man rides a bike", "a man on a bicycle", "a cyclist on the road"],
    ] * 4
    return captions


def test_cider_scores_match_and_prefer_correct_captions():
    corpus = _corpus()
    cider = Cider().fit(corpus)
    exact = cider.score(corpus[0][0], corpus[0])
    paraphrased = cider.score("a cat is sitting on a mat", corpus[0])
    unrelated = cider.score("a train passes a station", corpus[0])
    assert exact > paraphrased > unrelated
    assert exact <= 10.0 * cider.scale


def test_cider_batch_shape():
    corpus = _corpus()
    cider = Cider().fit(corpus)
    scores = cider.score_batch([c[0] for c in corpus], corpus)
    assert len(scores) == len(corpus)
    assert all(score >= 0 for score in scores)


def test_tokenizer():
    assert tokenize("A cat, on the mat!") == ["a", "cat", "on", "the", "mat"]


def test_vqa_accuracy_follows_the_official_metric():
    answers = ["cat"] * 3 + ["dog"] * 7
    # min(#annotators/3, 1) averaged over the ten annotators
    assert abs(vqa_accuracy("cat", answers) - 0.1) < 1e-9
    assert abs(vqa_accuracy("dog", answers) - 0.1) < 1e-9
    assert abs(vqa_accuracy("Cat", answers) - 0.1) < 1e-9  # normalization
    assert vqa_accuracy("bird", answers) == 0.0
    # five annotators out of ten agree -> min(5/3, 1) = 1 -> 1.0
    assert abs(vqa_accuracy("cat", ["cat"] * 5 + ["dog"] * 5) - 0.1) < 1e-9


def test_answer_normalization_removes_articles_and_punctuation():
    assert normalize_answer("The CATS!") == "cats"
    assert normalize_answer("a two") == "2"


def test_most_frequent_answers():
    answers = ["yes", "no", "yes", "maybe", "no", "no"]
    assert most_frequent_answers(answers, 2) == ["no", "yes"]


def test_pope_f1_and_parsing():
    predictions = ["Yes", "no", "Yes, there is"]
    labels = ["yes", "yes", "no"]
    # tp = 1, fp = 1, fn = 1 -> precision 1/2, recall 1/2, f1 = 0.5
    assert abs(pope_f1(predictions, labels) - 0.5) < 1e-9
    assert parse_yes_no("Answer: no.") == "no"
    assert parse_yes_no("The image shows a table") is None


def test_sqa_answer_parsing():
    assert parse_sqa_answer("The answer is B.") == "b"
    assert parse_sqa_answer("A: because...") == "a"
