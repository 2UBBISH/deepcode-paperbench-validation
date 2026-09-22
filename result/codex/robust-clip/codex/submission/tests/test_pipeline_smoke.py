"""End-to-end smoke tests of the FARE training step and the LVLM attack pipeline.

These tests use tiny random models (no downloads) so that they run in seconds on
a CPU.
"""

import torch
import torch.nn.functional as F

from robust_clip.attacks.lvlm_attack import EnsembleAttackConfig, LVLMEnsembleAttack
from robust_clip.models.lvlm.base import LVLM
from robust_clip.training.losses import fare_loss


class ToyLVLM(LVLM):
    """A differentiable stand-in for an LVLM: logits depend on the image mean."""

    def __init__(self, vocab=("yes", "no", "maybe", "maybe", "word")):
        super().__init__()
        self.vocab = list(vocab)
        self.weights = torch.randn(len(self.vocab), 3, generator=torch.Generator().manual_seed(0))
        self.fallback_index = 0

    def _index(self, text: str) -> int:
        lowered = text.strip().lower()
        return self.vocab.index(lowered) if lowered in self.vocab else self.fallback_index

    def _logits(self, images):
        pooled = images.mean(dim=(2, 3))  # [B, 3]
        return pooled @ self.weights.t()  # [B, V]

    def generate(self, images, prompts=None, max_new_tokens=4, num_beams=1, do_sample=False):
        indices = self._logits(images).argmax(dim=1).tolist()
        return [self.vocab[i] for i in indices]

    def target_loss(self, images, prompts, targets, reduction="none"):
        logits = self._logits(images)
        target_ids = torch.tensor([self._index(t) for t in targets])
        loss = F.cross_entropy(logits, target_ids, reduction="none")
        return loss if reduction == "none" else loss.mean()


def test_fare_training_step_reduces_the_loss():
    torch.manual_seed(0)
    theta = torch.randn(4, 6, requires_grad=True)
    x = torch.rand(2, 3, 8, 8)
    phi_org = torch.randn(2, 6, generator=torch.Generator().manual_seed(1))
    optimizer = torch.optim.AdamW([theta], lr=1e-2)

    def encode(image):
        return image.flatten(1)[:, :4] @ theta

    losses = []
    for _ in range(5):
        x_adv = (x + 0.02 * torch.sign(torch.randn_like(x))).clamp(0, 1)
        loss = fare_loss(encode(x_adv), encode(x), phi_org)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0]


def test_ensemble_attack_worst_case_score():
    model = ToyLVLM()
    images = torch.rand(2, 3, 8, 8)
    prompts = ["q"] * 2
    references = [["yes", "yes", "no", "no", "maybe"], ["no", "no", "no", "yes", "maybe"]]
    config = EnsembleAttackConfig(
        eps="4/255",
        half_precision_iters=2,
        single_precision_iters=2,
        num_ground_truths=3,
        score_threshold=0.0,
        max_new_tokens=2,
    )
    metric = lambda generations, refs: [float(gen in ref) for gen, ref in zip(generations, refs)]
    attack = LVLMEnsembleAttack(model, config, metric_fn=metric)
    result = attack.run(images, prompts, references, is_vqa=True)
    assert result.x_adv.shape == images.shape
    assert len(result.scores) == 2
    assert all(score <= clean for score, clean in zip(result.scores, result.clean_scores))
    assert (result.x_adv - images).abs().max() <= 4 / 255 + 1e-6


def test_fare_finetuner_runs_one_step_on_a_small_clip_model():
    """One FARE training step on a randomly initialized ViT-B/32 (CPU smoke test)."""
    from robust_clip.training.adv_train import AdversarialFineTuner, FineTuneConfig

    config = FineTuneConfig(
        method="fare",
        arch="ViT-B-32",
        pretrained=None,
        eps="4/255",
        pgd_steps=2,
        epochs=1,
        micro_batch_size=2,
        batch_size=2,
        device="cpu",
        log_every=1000,
    )
    trainer = AdversarialFineTuner(config)
    images = torch.rand(2, 3, 224, 224)
    labels = torch.tensor([0, 1])
    info = trainer.compute_loss(images, labels)
    assert torch.isfinite(info["loss"])
    assert info["loss"].requires_grad
    info["loss"].backward()
    grads = [p.grad for p in trainer.model.parameters() if p.requires_grad and p.grad is not None]
    assert len(grads) > 0
    # the frozen reference encoder must not receive gradients
    assert all(p.grad is None for p in trainer.ref_model.parameters())
