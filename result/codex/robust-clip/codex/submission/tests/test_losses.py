"""Unit tests of the FARE / TeCoA objectives and the embedding losses."""

import torch

from robust_clip.training.losses import (
    adversarial_embedding_loss,
    clean_embedding_loss,
    embedding_distance,
    fare_loss,
    tecoa_loss,
)


def test_embedding_distance_matches_definition():
    a = torch.tensor([[3.0, 4.0]])
    b = torch.zeros(1, 2)
    assert torch.isclose(embedding_distance(a, b, "l2_squared"), torch.tensor([25.0]))
    assert torch.isclose(embedding_distance(a, b, "l2"), torch.tensor([5.0]))


def test_fare_loss_is_adv_plus_lambda_times_clean():
    phi_org = torch.zeros(2, 4)
    phi_clean = torch.ones(2, 4)  # ||.||^2 = 4 per sample
    phi_adv = 2 * torch.ones(2, 4)  # ||.||^2 = 16 per sample
    assert torch.isclose(fare_loss(phi_adv, phi_clean, phi_org), torch.tensor(20.0))
    # with lambda = 0.5 only the adversarial term is halved
    assert torch.isclose(fare_loss(phi_adv, phi_clean, phi_org, clean_weight=0.5), torch.tensor(18.0))
    # the l1 ablation of App. B.4: ||2||_1 + ||1||_1 = 8 + 4
    assert torch.isclose(fare_loss(phi_adv, phi_clean, phi_org, norm="l1"), torch.tensor(12.0))


def test_fare_loss_does_not_require_grad_of_reference():
    phi_org = torch.zeros(1, 3)
    phi_clean = torch.zeros(1, 3, requires_grad=True)
    phi_adv = torch.zeros(1, 3, requires_grad=True)
    loss = fare_loss(phi_adv, phi_clean, phi_org)
    loss.backward()
    assert phi_clean.grad is not None and phi_adv.grad is not None
    assert phi_org.grad is None


def test_tecoa_loss_is_cross_entropy_on_adversarial_images():
    image_features = torch.eye(3)
    text_features = torch.eye(3)
    logit_scale = torch.tensor(1.0)
    loss = tecoa_loss(image_features, text_features, logit_scale)
    assert loss.item() < 0.6  # correct pairing -> small loss
    wrong = torch.roll(image_features, 1, dims=0)
    assert tecoa_loss(wrong, text_features, logit_scale).item() > loss.item()


def test_embedding_losses():
    phi_org = torch.zeros(2, 2)
    phi_clean = torch.full((2, 2), 1.0)
    phi_adv = torch.full((2, 2), 3.0)
    assert torch.isclose(clean_embedding_loss(phi_clean, phi_org, reduction="mean"), torch.tensor(2.0))
    assert torch.isclose(adversarial_embedding_loss(phi_adv, phi_org, reduction="mean"), torch.tensor(18.0))
