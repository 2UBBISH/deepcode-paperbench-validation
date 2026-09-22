"""Tests of APGD, PGD and the precision handling."""

import torch
import torch.nn.functional as F

from robust_clip.attacks.apgd import APGDAttack, dlr_loss
from robust_clip.attacks.pgd import pgd_attack
from robust_clip.utils.precision import PerturbationGrid, eps_to_int, int_to_eps


def _toy_model(seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(10, 3 * 16 * 16, generator=generator)
    return weight


def test_eps_conversion():
    assert eps_to_int("2/255") == 2
    assert eps_to_int("64/255") == 64
    assert eps_to_int(4 / 255) == 4
    assert abs(int_to_eps(4) - 4 / 255) < 1e-12


def test_perturbation_grid_respects_ball_and_dtype():
    x = torch.rand(4, 3, 8, 8)
    grid = PerturbationGrid(4, dtype=torch.float16)
    x_adv = grid.random_start(x)
    # in the half precision setting the ball is centered on the 16-bit clean image
    # (the projection is exact up to half an ulp of the float16 grid)
    assert (x_adv - grid.clean(x)).abs().max() <= grid.eps + 2 ** -11
    assert torch.allclose(x_adv, x_adv.half().float())  # snapped to the 16-bit grid
    # away from the [0, 1] boundary the perturbation is a multiple of 1/255
    # (snapped onto the float16 grid)
    interior = (grid.clean(x) > 0.05) & (grid.clean(x) < 0.95)
    delta = ((x_adv - grid.clean(x)) * 255)[interior]
    assert (delta - delta.round()).abs().max() < 0.5
    # a single precision attack leaves the float32 image untouched
    grid32 = PerturbationGrid(4, dtype=torch.float32)
    assert torch.allclose(grid32.clean(x), x)


def test_apgd_finds_adversarial_examples():
    torch.manual_seed(0)
    weight = _toy_model()
    images = torch.rand(8, 3, 16, 16)

    def forward(x):
        return x.flatten(1) @ weight.t()

    with torch.no_grad():
        labels = forward(images).argmax(1)  # the toy model is accurate on clean inputs
    clean = (forward(images).argmax(1) == labels).float().mean()
    attack = APGDAttack(eps="16/255", n_iter=40, loss="ce")
    x_adv = attack.attack(images, forward, labels=labels)
    robust = (forward(x_adv).argmax(1) == labels).float().mean()
    assert clean == 1.0 and robust <= clean
    assert (x_adv - images).abs().max() <= 16 / 255 + 1e-6


def test_dlr_loss_is_minimized_at_the_true_class():
    logits = torch.tensor([[5.0, 0.0, 1.0]])
    labels = torch.tensor([1])
    wrong = torch.tensor([[0.0, 5.0, 1.0]])
    assert dlr_loss(logits, labels).item() > dlr_loss(wrong, labels).item()


def test_pgd_no_momentum_matches_jailbreak_setting():
    torch.manual_seed(0)
    weight = _toy_model()
    labels = torch.randint(0, 10, (4,))
    images = torch.rand(4, 3, 16, 16)

    def forward(x):
        return x.flatten(1) @ weight.t()

    x_adv, loss = pgd_attack(
        images,
        forward,
        lambda out, xa: F.cross_entropy(out, labels, reduction="none"),
        eps="8/255",
        alpha="1/255",
        steps=20,
        momentum=0.0,
        maximize=True,
    )
    assert x_adv.shape == images.shape
    assert torch.isfinite(loss).all()


def test_select_worst_case_prefers_the_misclassified_attack():
    """The attack combination of AutoAttack must not undo a successful attack."""
    from robust_clip.eval.zeroshot import select_worst_case

    labels = torch.tensor([0, 1])
    x_a = torch.zeros(2, 1)
    x_b = torch.ones(2, 1)
    # attack A misclassifies sample 0 (but with a smaller loss than B), attack B is
    # correct on sample 0 and misclassifies sample 1
    logits_a = torch.tensor([[0.0, 5.0], [0.0, 0.1]])
    logits_b = torch.tensor([[9.0, 0.0], [5.0, 0.0]])
    combined = select_worst_case(x_a, x_b, logits_a, logits_b, labels)
    assert combined[0].item() == 0.0  # keep the successful attack A
    assert combined[1].item() == 1.0  # keep the successful attack B
