"""Fast smoke tests: they exercise the paper's building blocks without CLIP/LLaVA.

Run with ``pytest tests/ -q`` (or directly with ``python tests/test_smoke.py``).
None of the tests needs a GPU or a network connection.
"""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robust_clip.attacks.apgd import (  # noqa: E402
    APGDAttack,
    dlr_loss,
    dlr_loss_targeted,
    make_loss,
    runner_up_classes,
)
from robust_clip.attacks.lvlm_attack import LVLMAttackPipeline  # noqa: E402
from robust_clip.eval.cider import CiderScorer  # noqa: E402
from robust_clip.eval.metrics import pope_f1, vqa_accuracy, vqa_score  # noqa: E402
from robust_clip.lvlm.base import LVLM  # noqa: E402
from robust_clip.training.losses import FARELoss, TeCoALoss  # noqa: E402
from robust_clip.training.pgd import pgd_linf_maximize  # noqa: E402
from robust_clip.training.train import build_scheduler  # noqa: E402
from robust_clip.theory import cosine_difference, theorem_3_1_bound  # noqa: E402
from robust_clip.utils.common import human_readable_epsilon, parse_epsilon  # noqa: E402
from robust_clip.utils.quant import quantize_pixels  # noqa: E402


# --------------------------------------------------------------------------- #
#                                   helpers                                    #
# --------------------------------------------------------------------------- #
class ToyEncoder(nn.Module):
    """Stand-in for the CLIP image encoder: a small CNN with an ``encode_image``."""

    def __init__(self, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.conv = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1)
        self.head = nn.Linear(8, 16)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        features = F.relu(self.conv(images))
        features = features.mean(dim=(2, 3))
        return self.head(features)


class ToyClassifier(nn.Module):
    """A tiny linear model used to check that APGD actually breaks it."""

    def __init__(self, dim: int = 3 * 8 * 8, classes: int = 4, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.fc = nn.Linear(dim, classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.fc(images.flatten(1))


class DummyLVLM(LVLM):
    """Minimal LVLM for the attack-pipeline tests.

    The "model" maps the mean pixel value of an image to two logits; the
    continuation ``"yes"`` corresponds to class 0 and any other continuation to
    class 1.  The attack can therefore be checked by looking at the answer the
    dummy model would produce.
    """

    def __init__(self, seed: int = 0):
        torch.manual_seed(seed)
        # score = weight * mean(pixels) + bias; clean images in [0.4, 0.6] are
        # classified as "yes" while a perturbation of ~0.1 flips the answer.
        self.weight = nn.Parameter(torch.tensor(5.0))
        self.bias = nn.Parameter(torch.tensor(-1.75))
        self.name = "dummy"
        self._dtype = torch.float32
        self._device = torch.device("cpu")
        self.generated = 0

    def parameters(self):
        return iter([self.weight, self.bias])

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    def set_precision(self, dtype: torch.dtype) -> None:
        self._dtype = dtype

    def to(self, device):
        self._device = torch.device(device)
        return self

    def _logits(self, images: torch.Tensor) -> torch.Tensor:
        mean = images.flatten(1).mean(dim=1, keepdim=True)
        score = self.weight.to(images.dtype) * mean + self.bias.to(images.dtype)
        return torch.cat([score, -score], dim=1)

    def nll(self, images, prompts, continuations, reduction="mean"):
        targets = torch.tensor(
            [0 if c.strip().lower() == "yes" else 1 for c in continuations], device=images.device
        )
        return F.cross_entropy(self._logits(images), targets, reduction="none")

    def generate(self, images, prompts, max_new_tokens: int = 8, **kwargs):
        self.generated += 1
        preds = self._logits(images).argmax(dim=1)
        return ["yes" if int(p) == 0 else "no" for p in preds]

    def preprocess(self, images):
        raise NotImplementedError


# --------------------------------------------------------------------------- #
#                                     tests                                    #
# --------------------------------------------------------------------------- #
def test_epsilon_parsing():
    assert abs(parse_epsilon("2/255") - 2 / 255) < 1e-12
    assert abs(parse_epsilon("4/255") - 4 / 255) < 1e-12
    assert abs(parse_epsilon(0.1) - 0.1) < 1e-12
    assert human_readable_epsilon(2 / 255) == "2/255"


def test_quantization_grid():
    images = torch.rand(2, 3, 8, 8)
    out16 = quantize_pixels(images, 16)
    assert out16.shape == images.shape
    assert torch.allclose(out16 * 255, torch.round(out16 * 255), atol=1e-4)
    assert float(out16.max()) <= 1.0 and float(out16.min()) >= 0.0


def test_pgd_inner_maximisation_respects_ball():
    x = torch.rand(4, 3, 8, 8)
    target = torch.zeros_like(x)

    def objective(x_adv):
        # maximise the distance to the clean image
        return (x_adv - x).pow(2).flatten(1).sum(dim=1)

    eps = 4 / 255
    x_adv, losses = pgd_linf_maximize(
        objective, x, eps=eps, alpha=1 / 255, n_steps=10, momentum=0.9
    )
    assert (x_adv - x).abs().max() <= eps + 1e-6
    assert x_adv.min() >= 0.0 and x_adv.max() <= 1.0
    clean_loss = objective(x)
    assert (losses >= clean_loss - 1e-6).all()
    del target


def test_fare_loss_matches_definition():
    encoder = ToyEncoder()
    reference = ToyEncoder(seed=1)
    loss_fn = FARELoss(
        encode_fn=encoder.encode_image,
        reference_encoder=reference,
        eps=2 / 255,
        alpha=1 / 255,
        n_steps=3,
        keep_best=True,
    )
    x = torch.rand(2, 3, 8, 8)
    loss, x_adv = loss_fn(x)
    assert loss.dim() == 0 and loss.item() >= 0.0
    assert (x_adv - x).abs().max() <= 2 / 255 + 1e-6

    # the reference embedding is the *clean* one and is not differentiated
    reference_embedding = loss_fn.reference_embedding(x)
    manual = (encoder.encode_image(x_adv) - reference_embedding).pow(2).sum(dim=1).mean()
    assert torch.allclose(loss, manual, atol=1e-5)


def test_fare_loss_is_minimised_by_training_step():
    encoder = ToyEncoder()
    reference = ToyEncoder(seed=3)
    loss_fn = FARELoss(
        encode_fn=encoder.encode_image, reference_encoder=reference, eps=2 / 255, alpha=1 / 255, n_steps=1
    )
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-2)
    x = torch.rand(4, 3, 8, 8)
    first, _ = loss_fn(x)
    for _ in range(10):
        loss, _ = loss_fn(x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    last, _ = loss_fn(x)
    assert last.item() < first.item()


def test_tecoa_loss_uses_cosine_similarity():
    encoder = ToyEncoder()
    text = F.normalize(torch.randn(5, 16), dim=-1)
    loss_fn = TeCoALoss(
        encode_fn=encoder.encode_image, text_embeddings=text, eps=2 / 255, alpha=1 / 255, n_steps=2
    )
    x = torch.rand(3, 3, 8, 8)
    y = torch.randint(0, 5, (3,))
    loss, x_adv = loss_fn(x, y)
    assert loss.item() > 0
    z = F.normalize(encoder.encode_image(x_adv), dim=-1)
    manual = F.cross_entropy(100.0 * (z @ text.t()), y)
    assert torch.allclose(loss, manual, atol=1e-4)


def test_apgd_breaks_a_toy_classifier():
    model = ToyClassifier()
    x = torch.rand(32, 3, 8, 8)
    y = model(x).argmax(dim=1)
    clean_acc = float((model(x).argmax(dim=1) == y).float().mean())
    assert clean_acc == 1.0

    loss_fn = make_loss("ce", targets=y, logits_fn=model)
    attack = APGDAttack(eps=0.2, n_iter=40, alpha=0.2, n_restarts=1)
    x_adv, _ = attack.perturb(loss_fn, x, maximize=True)
    adv_acc = float((model(x_adv).argmax(dim=1) == y).float().mean())
    assert adv_acc < 0.5
    assert (x_adv - x).abs().max() <= 0.2 + 1e-6


def test_theorem_3_1_holds():
    """The embedding distance really bounds the change of the cosine similarity."""
    torch.manual_seed(0)
    for scale, distance in [(1.0, 0.05), (10.0, 0.2), (0.5, 0.01)]:
        phi_org = torch.randn(64, 32) * scale
        direction = torch.randn(64, 32)
        direction = direction / direction.norm(dim=-1, keepdim=True)
        phi_ft = phi_org + distance * direction
        psi = torch.randn(10, 32)
        lhs = cosine_difference(phi_org, phi_ft, psi).max(dim=1).values
        rhs = theorem_3_1_bound(phi_org, phi_ft)
        assert (lhs <= rhs + 1e-5).all(), f"bound violated for scale={scale}"
        # and the bound is tight enough to be informative
        assert (rhs > 0).all()


def test_dlr_losses():
    logits = torch.tensor([[3.0, 1.0, 0.5, -1.0], [0.2, 2.5, 1.0, 0.0]])
    y = torch.tensor([0, 1])
    targets = torch.tensor([1, 2])
    assert dlr_loss(logits, y).shape == (2,)
    assert dlr_loss_targeted(logits, y, targets).shape == (2,)
    assert runner_up_classes(logits, y).tolist() == [1, 2]


def test_lr_schedule_warmup_and_cosine():
    param = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([param], lr=1.0)
    scheduler = build_scheduler(optimizer, total_steps=100, warmup_frac=0.07, schedule="cosine")
    lrs = []
    for _ in range(100):
        lrs.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()
    assert lrs[0] < lrs[6]                      # warm-up
    assert abs(lrs[7] - 1.0) < 1e-6             # peak at ~7% of the run
    assert lrs[-1] < 0.05                       # cosine decay to ~0


def test_cider_prefers_the_matching_caption():
    scorer = CiderScorer(scale=100.0)
    references = [
        ["a cat is sitting on a mat", "the cat sits on the mat", "a cat on a mat"],
        ["a dog runs in the park", "a dog is running outside", "a dog in a field"],
    ]
    scorer.prepare(references)
    good = scorer.score("a cat is sitting on a mat", references[0])
    bad = scorer.score("a train is leaving the station", references[0])
    assert good > bad


def test_vqa_and_pope_metrics():
    answers = ["yes"] * 10
    assert abs(vqa_score("yes", answers) - 1.0) < 1e-6
    assert vqa_score("no", answers) == 0.0
    # three annotators out of ten said "maybe" -> min(3/3, 1)
    assert abs(vqa_score("maybe", ["maybe"] * 3 + ["no"] * 7) - 1.0) < 1e-6
    assert abs(vqa_score("maybe", ["maybe"] + ["no"] * 9) - 1 / 3) < 1e-6
    scores = vqa_accuracy(["yes", "no"], [answers, answers])
    assert abs(scores[0] - 1.0) < 1e-6 and scores[1] == 0.0

    metrics = pope_f1(["Yes", "No", "no", "yes"], ["yes", "yes", "no", "no"])
    assert abs(metrics["precision"] - 0.5) < 1e-6
    assert abs(metrics["recall"] - 0.5) < 1e-6


def test_lvlm_pipeline_runs_and_stays_in_the_ball():
    lvlm = DummyLVLM()
    images = torch.rand(6, 3, 8, 8) * 0.2 + 0.4
    prompts = ["USER: <image>\nis there a cat?\nASSISTANT:"] * 6
    answers = [["yes"] * 10] * 6
    eps = 0.2

    def score_fn(predictions, _answers):
        return vqa_accuracy(predictions, answers)

    pipeline = LVLMAttackPipeline(
        lvlm,
        eps=eps,
        n_iter=10,
        alpha=eps,
        use_half_precision_stage=True,
        use_single_precision_stage=True,
        max_new_tokens=4,
    )
    result = pipeline.attack_vqa(
        images, prompts, answers, score_fn, max_answers=1, targeted_strings=("maybe",)
    )
    assert result.images.shape == images.shape
    max_delta = float((result.images - images).abs().max())
    assert max_delta <= eps + 1e-6, f"perturbation {max_delta} leaves the ball"
    assert result.scores.shape == (6,)
    assert result.scores.max() <= 1.0
    # the attacker should have found images that are not classified as "yes"
    assert result.scores.mean() < 1.0, f"attack failed: scores={result.scores.tolist()}"


if __name__ == "__main__":  # pragma: no cover
    failures = []
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures.append((name, exc))
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(failures)} failing test(s)")
    raise SystemExit(1 if failures else 0)
