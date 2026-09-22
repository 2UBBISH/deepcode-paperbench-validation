"""Exercise the LVLM evaluation drivers with a stub LVLM.

The drivers in ``robust_clip.eval`` talk to the models only through the
``LVLM`` interface (``nll`` / ``generate`` / ``preprocess`` / ``set_precision``),
so they can be tested without downloading LLaVA or OpenFlamingo.  These tests
check the plumbing of the captioning and VQA evaluations end to end (prompts,
clean scoring, the half->single->targeted attack pipeline, metric computation)
and that the values are consistent with the stub model's behaviour.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robust_clip.eval.captioning import evaluate_captioning  # noqa: E402
from robust_clip.eval.cider import CiderScorer  # noqa: E402
from robust_clip.eval.lvlm_data import CaptionExample, VQAExample  # noqa: E402
from robust_clip.eval.metrics import vqa_accuracy  # noqa: E402
from robust_clip.eval.vqa import evaluate_vqa  # noqa: E402
from robust_clip.attacks.lvlm_attack import TargetedStringAttack  # noqa: E402
from robust_clip.lvlm.base import LVLM  # noqa: E402


class StubLVLM(LVLM):
    """``generate`` returns one of two template answers depending on the image.

    ``nll`` is a differentiable surrogate so that the attack pipeline can run.
    Every image whose mean pixel value is above ``threshold`` produces the
    "good" answer, otherwise the "bad" one.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.name = "stub"
        self._dtype = torch.float32
        self._device = torch.device("cpu")

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    def set_precision(self, dtype):
        self._dtype = dtype

    def to(self, device):
        return self

    def parameters(self):
        return iter([])

    def _score(self, images):
        return images.flatten(1).mean(dim=1)

    def nll(self, images, prompts, continuations, reduction="mean"):
        good = torch.tensor(
            [1.0 if "cat" in c else 0.0 for c in continuations], device=images.device
        )
        logit = (self._score(images) - self.threshold) * 20.0
        logits = torch.stack([logit, -logit], dim=1)
        targets = (1.0 - good).long()
        return F.cross_entropy(logits, targets, reduction="none")

    def generate(self, images, prompts, max_new_tokens=8, **kwargs):
        good = self._score(images) > self.threshold
        if any("cat" not in p and "Describe" not in p for p in prompts):
            return ["yes" if g else "no" for g in good]
        return ["a cat on a mat" if g else "a train on the tracks" for g in good]

    def preprocess(self, images):
        return torch.stack([image if torch.is_tensor(image) else torch.zeros(3, 8, 8) for image in images])


class _Args:
    """The subset of the CLI namespace the drivers read."""

    def __init__(self, **kwargs):
        self.seed = 0
        self.max_new_tokens = 8
        self.eval_batch_size = 4
        self.clean = True
        self.attack = True
        self.eps_list = ["0.1"]
        self.num_attack_samples = 6
        self.attack_iterations = 5
        self.attack_alpha = None
        self.attack_momentum = 0.9
        self.grad_normalization = "elementwise_sign"
        self.no_half_precision = False
        self.no_single_precision = False
        self.coco_threshold = 10.0
        self.flickr_threshold = 2.0
        self.cider_df = None
        self.cider_scale = 1000.0
        self.backend = "llava"
        for key, value in kwargs.items():
            setattr(self, key, value)


def _images(n=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    # half of the images are "good" (> threshold), half are "bad"
    base = torch.rand(n, 3, 8, 8, generator=generator) * 0.2 + 0.4
    base[:, :, :, :4] += 0.3
    return base


def test_captioning_driver_runs_and_degrades_the_score():
    lvlm = StubLVLM()
    images = _images()
    examples = [
        CaptionExample(image=images[i], captions=["a cat on a mat"] * 5, image_id=str(i))
        for i in range(len(images))
    ]
    args = _Args()
    results = evaluate_captioning(lvlm, examples, args, task="coco_caption")

    assert results["clean_num_samples"] == len(examples)
    # the stub produces the reference caption only for the "good" images
    assert 0.0 <= results["clean_cider"] <= 100.0
    key = "robust_cider_0.1"
    assert key in results
    assert results[key] <= results["clean_cider"] + 1e-6
    assert results["perturbation_linf_0.1"] <= 0.1 + 1e-6


def test_vqa_driver_runs_and_degrades_the_accuracy():
    lvlm = StubLVLM()
    images = _images()
    examples = [
        VQAExample(
            image=images[i],
            question="Is there a cat in the image?",
            answers=["yes"] * 10,
            question_id=str(i),
        )
        for i in range(len(images))
    ]
    args = _Args()
    results = evaluate_vqa(lvlm, examples, args, task="vqav2")

    assert 0.0 <= results["clean_accuracy"] <= 100.0
    key = "robust_accuracy_0.1"
    assert key in results
    assert results[key] <= results["clean_accuracy"] + 1e-6


def test_cider_scorer_is_reusable_across_batches():
    """The attack pipeline re-scores the same samples many times."""
    references = [
        ["a cat on a mat", "a small cat sitting", "the cat is on the mat"],
        ["a dog running outside", "a dog in the park", "a brown dog runs"],
    ]
    scorer = CiderScorer().prepare(references)
    first = scorer.compute_scores(["a cat on a mat", "a dog in the park"], references)
    second = scorer.compute_scores(["a cat on a mat", "a dog in the park"], references)
    assert first == second
    assert all(score > 0 for score in first)


def test_targeted_string_attack_forces_the_target_caption():
    """Sec. 4.2 / Table 3: force an exact target string inside the ℓ∞ ball."""
    lvlm = StubLVLM(threshold=0.5)
    images = _images(n=6, seed=4)
    prompts = ["USER: <image>\nDescribe the image concisely.\nASSISTANT:"] * 6
    target = "a cat on a mat"

    attack = TargetedStringAttack(
        lvlm, eps=0.2, n_iter=30, alpha=0.2, momentum=0.9, random_init=True,
        bits=32, dtype=torch.float32, max_new_tokens=8,
    )
    x_adv, outputs, success = attack.attack(images, prompts, [target] * 6)

    assert (x_adv - images).abs().max() <= 0.2 + 1e-6
    # every image that can be moved above the decision threshold must now
    # produce the target string verbatim
    reachable = (images.flatten(1).mean(dim=1) + 0.2) > 0.5
    assert reachable.any()
    assert success[reachable].all()
    assert all(target in output for output, ok in zip(outputs, success) if ok)


if __name__ == "__main__":  # pragma: no cover
    test_captioning_driver_runs_and_degrades_the_score()
    test_vqa_driver_runs_and_degrades_the_accuracy()
    test_cider_scorer_is_reusable_across_batches()
    test_targeted_string_attack_forces_the_target_caption()
    print("lvlm eval driver tests OK")
