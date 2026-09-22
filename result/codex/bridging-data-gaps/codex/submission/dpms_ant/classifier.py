"""The binary source/target domain classifier ``p_phi``.

The classifier is the *similarity measure* of the paper.  Section 5.2 plus the
supplementary material specify how it is obtained:

* start from a pre-trained 256x256 ImageNet classifier
  (``256x256_classifier.pt``, Dhariwal & Nichol 2021),
* "modify the last layer to output two classes to classify whether images were
  coming from the source or the target dataset",
* fine-tune with Adam, learning rate ``1e-4``, batch size ``64`` for ``300``
  iterations,
* train on *noised* images ``x_t`` at random timesteps ``t`` -- "the
  classifiers being trained on noised targeted images among T (1000 steps) as
  Equation (1), ensuring a robust gradient for training" -- which is why only
  10 target images suffice (Section 5.5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .schedules import DiffusionSchedule
from .utils import BatchIterator, Logger, ensure_dir, progbar


@dataclass
class ClassifierTrainConfig:
    """Hyper-parameters from the supplementary material (Section 5.2)."""

    iterations: int = 300
    lr: float = 1e-4
    batch_size: int = 64
    betas: tuple = (0.9, 0.999)
    weight_decay: float = 0.0
    freeze_backbone: bool = False  # True = train only the new binary head
    log_every: int = 50
    seed: int = 0


def build_classifier_optimizer(
    classifier: nn.Module, config: ClassifierTrainConfig
) -> torch.optim.Optimizer:
    if config.freeze_backbone:
        params = [p for p in classifier.parameters() if p.requires_grad]
    else:
        params = list(classifier.parameters())
    return torch.optim.Adam(
        params, lr=config.lr, betas=config.betas, weight_decay=config.weight_decay
    )


def classifier_step(
    classifier: nn.Module,
    schedule: DiffusionSchedule,
    x_start: torch.Tensor,
    labels: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> tuple:
    """One supervised step on noised images (Eq. 1 sampling + cross entropy)."""
    batch = x_start.shape[0]
    device = x_start.device
    timesteps = torch.randint(
        0, schedule.num_timesteps, (batch,), device=device, generator=generator
    ).long()
    noise = torch.randn(x_start.shape, device=device, dtype=x_start.dtype, generator=generator)
    x_t = schedule.q_sample(x_start, timesteps, noise)
    logits = classifier(x_t, timesteps)
    loss = F.cross_entropy(logits.float(), labels)
    predictions = logits.argmax(dim=-1)
    accuracy = (predictions == labels).float().mean()
    return loss, accuracy


def train_domain_classifier(
    classifier: nn.Module,
    schedule: DiffusionSchedule,
    batches: Callable[[], tuple],
    config: Optional[ClassifierTrainConfig] = None,
    device: str = "cpu",
    logger: Optional[Logger] = None,
) -> Dict[str, float]:
    """Fine-tune ``p_phi`` to separate source from target images.

    :param batches: infinite iterator yielding ``(x_start, labels)`` batches
        with ``labels = 0`` for source and ``1`` for target images.
    :returns: history dictionary (loss/accuracy curves).
    """
    config = config or ClassifierTrainConfig()
    classifier = classifier.to(device)
    if config.freeze_backbone:
        # keep the pre-trained representation, learn only the new 2-way head
        for name, parameter in classifier.named_parameters():
            parameter.requires_grad_("out" in name)
    optimizer = build_classifier_optimizer(classifier, config)
    history: Dict[str, List[float]] = {"loss": [], "accuracy": []}

    classifier.train()
    batches = BatchIterator(batches)
    for step in progbar(range(config.iterations), desc="classifier"):
        x_start, labels = next(batches)
        x_start = x_start.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device).long()
        loss, accuracy = classifier_step(classifier, schedule, x_start, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history["loss"].append(float(loss.detach()))
        history["accuracy"].append(float(accuracy.detach()))
        if logger is not None and (step % config.log_every == 0 or step == config.iterations - 1):
            logger.log(
                f"[classifier] step {step:5d} loss {float(loss):.4f} acc {float(accuracy):.3f}"
            )
    history["final_loss"] = history["loss"][-1] if history["loss"] else float("nan")
    history["final_accuracy"] = history["accuracy"][-1] if history["accuracy"] else float("nan")
    classifier.eval()
    return history


def save_classifier(classifier: nn.Module, path: str, config: Optional[ClassifierTrainConfig] = None) -> str:
    ensure_dir(path.rsplit("/", 1)[0] if "/" in path else ".")
    torch.save({"model": classifier.state_dict(), "config": config}, path)
    return path


def load_classifier(path: str, map_location: str = "cpu") -> nn.Module:
    payload = torch.load(path, map_location=map_location)
    return payload["model"] if isinstance(payload, dict) and "model" in payload else payload


# ---------------------------------------------------------------------- #
# toy (2-D) classifier used by the Section 5.1 experiment
# ---------------------------------------------------------------------- #
class ToyDomainClassifier(nn.Module):
    """Small MLP classifier of noised 2-D samples, used for the toy study.

    Implements ``p_phi(y | x_t)`` for the toy setup: the timestep is encoded
    with the standard sinusoidal embedding and concatenated to ``x_t``.
    """

    def __init__(
        self,
        hidden: int = 128,
        depth: int = 3,
        num_classes: int = 2,
        time_dim: int = 16,
        data_dim: int = 2,
    ):
        super().__init__()
        self.time_dim = time_dim
        layers: List[nn.Module] = []
        in_features = data_dim + time_dim
        for _ in range(depth):
            layers += [nn.Linear(in_features, hidden), nn.SiLU()]
            in_features = hidden
        layers += [nn.Linear(hidden, num_classes)]
        self.net = nn.Sequential(*layers)

    @staticmethod
    def _timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: float = 10000.0):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32, device=timesteps.device)
            / half
        )
        args = timesteps.float().reshape(-1, 1) * freqs.reshape(1, -1)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        embedding = self._timestep_embedding(timesteps, self.time_dim).to(x.dtype)
        return self.net(torch.cat([x, embedding], dim=-1))


def train_toy_classifier(
    classifier: ToyDomainClassifier,
    schedule: DiffusionSchedule,
    source_samples: torch.Tensor,
    target_samples: torch.Tensor,
    batch_size: int = 64,
    iterations: int = 300,
    lr: float = 1e-4,
    device: str = "cpu",
    seed: int = 0,
) -> ToyDomainClassifier:
    """Same recipe as :func:`train_domain_classifier`, for the 2-D toy data."""
    generator = torch.Generator().manual_seed(seed)
    source = source_samples.to(device)
    target = target_samples.to(device)

    def batches():
        while True:
            index_s = torch.randint(0, source.shape[0], (batch_size // 2,), generator=generator)
            index_t = torch.randint(0, target.shape[0], (batch_size - batch_size // 2,), generator=generator)
            x = torch.cat([source[index_s], target[index_t]], dim=0)
            y = torch.cat(
                [
                    torch.zeros(batch_size // 2, dtype=torch.long),
                    torch.ones(batch_size - batch_size // 2, dtype=torch.long),
                ]
            )
            yield x, y

    train_domain_classifier(
        classifier,
        schedule,
        batches,
        ClassifierTrainConfig(iterations=iterations, lr=lr, batch_size=batch_size, seed=seed),
        device=device,
    )
    return classifier
