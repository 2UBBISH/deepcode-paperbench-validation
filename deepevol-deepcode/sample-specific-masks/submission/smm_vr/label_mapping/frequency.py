"""Frequency distribution of predicted labels.

Implements Algorithm 2 of the SMM paper (ICML 2024):

    Algorithm 2  Computing Frequency Distribution of
                 [f_P(f_in(x_i | theta)), y^T]
    Input : Target training set {(x_i^T, y_i^T)}_{i=1..n}, given input VR
            f_in(. | theta) and pre-trained model f_P(.)
    Output: Frequency distribution matrix d in Z^{|Y^P| x |Y^T|}
    Initialize d <- {0}^{|Y^P| x |Y^T|}
    for i = 1 ... n do
        y_hat_i^P <- f_P(f_in(x_i^T | theta))
        d_{y_hat_i^P, y_i^T} <- d_{y_hat_i^P, y_i^T} + 1
    end for

The matrix ``d`` counts, for every target class ``y^T``, how many target
training samples are predicted as each pre-trained (ImageNet) class ``y^P``.
It is the shared building block of both frequent label mapping (Flm,
Algorithm 3) and iterative label mapping (Ilm, Algorithm 4); it carries no
learnable parameters.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import torch

__all__ = [
    "compute_frequency_matrix",
    "frequency_matrix_from_predictions",
    "collect_predictions",
    "zero_frequency_matrix",
]


Tensor = torch.Tensor
PredictFn = Callable[[Tensor], Tensor]


def zero_frequency_matrix(
    num_pretrained_classes: int,
    num_target_classes: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.long,
) -> Tensor:
    """Create the ``d <- {0}^{|Y^P| x |Y^T|}`` matrix of Algorithm 2."""
    if num_pretrained_classes <= 0 or num_target_classes <= 0:
        raise ValueError("class counts must be positive")
    return torch.zeros(
        (num_pretrained_classes, num_target_classes), dtype=dtype, device=device
    )


def frequency_matrix_from_predictions(
    predicted: Tensor,
    targets: Tensor,
    num_pretrained_classes: int,
    num_target_classes: int,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Accumulate ``d[y_hat^P, y^T] += 1`` for a batch of predictions.

    Parameters
    ----------
    predicted:
        Predicted pre-trained-space labels ``y_hat_i^P`` (any shape; flattened).
    targets:
        Ground-truth target labels ``y_i^T`` (any shape; flattened).
    """
    predicted = torch.as_tensor(predicted).reshape(-1).to(torch.long).cpu()
    targets = torch.as_tensor(targets).reshape(-1).to(torch.long).cpu()
    if predicted.numel() != targets.numel():
        raise ValueError(
            f"predictions ({predicted.numel()}) and targets ({targets.numel()}) "
            "must have the same number of elements"
        )
    d = zero_frequency_matrix(
        num_pretrained_classes, num_target_classes, device=torch.device("cpu")
    )
    if predicted.numel() == 0:
        return d if device is None else d.to(device)
    # Guard against out-of-range labels coming from a modified classifier.
    valid = (
        (predicted >= 0)
        & (predicted < num_pretrained_classes)
        & (targets >= 0)
        & (targets < num_target_classes)
    )
    predicted, targets = predicted[valid], targets[valid]
    flat = predicted * num_target_classes + targets
    counts = torch.bincount(flat, minlength=num_pretrained_classes * num_target_classes)
    d = counts.reshape(num_pretrained_classes, num_target_classes)
    return d if device is None else d.to(device)


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    data_loader,
    f_in: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """Run ``f_P(f_in(x_i^T))`` over a data loader and return argmax labels.

    Returns
    -------
    (predicted, targets):
        ``predicted`` holds the arg-max pre-trained label ``y_hat_i^P`` and
        ``targets`` the ground-truth target label ``y_i^T`` (both 1-D long).
    """
    if device is None:
        device = next(model.parameters()).device
    was_training_model = model.training
    model.eval()
    if f_in is not None:
        f_in.eval()

    preds, labels = [], []
    for step, batch in enumerate(data_loader):
        if max_batches is not None and step >= max_batches:
            break
        if isinstance(batch, (list, tuple)):
            images, target = batch[0], batch[1]
        else:
            images, target = batch, None
        images = images.to(device, non_blocking=True)
        if f_in is not None:
            images = f_in(images)
        logits = model(images)
        preds.append(logits.argmax(dim=1).detach().cpu())
        if target is not None:
            labels.append(torch.as_tensor(target).reshape(-1).cpu())

    if was_training_model:
        model.train()

    predicted = (
        torch.cat(preds) if preds else torch.zeros(0, dtype=torch.long)
    )
    target = (
        torch.cat(labels) if labels else torch.zeros(0, dtype=torch.long)
    )
    return predicted, target


def compute_frequency_matrix(
    model: torch.nn.Module,
    data_loader,
    num_target_classes: int,
    num_pretrained_classes: Optional[int] = None,
    f_in: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
    predicted: Optional[Tensor] = None,
    targets: Optional[Tensor] = None,
) -> Tensor:
    """Algorithm 2: frequency distribution matrix ``d in Z^{|Y^P| x |Y^T|}``.

    Either pass a ``(predicted, targets)`` pair through ``predicted`` /
    ``targets``, or let the function forward all target training samples
    through ``f_P(f_in(x | theta))`` and take the arg-max prediction.

    Parameters
    ----------
    model:
        Frozen pre-trained classifier ``f_P``, returning logits over the
        pre-trained label space (ImageNet-1K by default).
    data_loader:
        Iterable of ``(images, target)`` over the target training set.
    num_target_classes:
        ``|Y^T|``, the number of classes of the target task.
    num_pretrained_classes:
        ``|Y^P|``; defaults to the model's output dimensionality.
    f_in:
        Input reprogramming function ``f_in(. | theta)``. ``None`` means the
        identity function (``theta <- 0``, the initialization used by Flm).
    """
    if num_pretrained_classes is None:
        out_features = getattr(model, "out_features", None)
        if out_features is None:
            out_features = getattr(getattr(model, "head", None), "out_features", None)
        if out_features is None:
            # torchvision ViT uses ``heads.head``
            heads = getattr(model, "heads", None)
            out_features = getattr(getattr(heads, "head", None), "out_features", None)
        if out_features is None:
            raise ValueError(
                "cannot infer |Y^P| from the model; pass num_pretrained_classes"
            )
        num_pretrained_classes = int(out_features)

    if predicted is None:
        predicted, targets = collect_predictions(
            model,
            data_loader,
            f_in=f_in,
            device=device,
            max_batches=max_batches,
        )
    if targets is None:
        raise ValueError("`targets` must be provided when `predicted` is given")

    return frequency_matrix_from_predictions(
        predicted,
        targets,
        num_pretrained_classes=num_pretrained_classes,
        num_target_classes=num_target_classes,
    )
