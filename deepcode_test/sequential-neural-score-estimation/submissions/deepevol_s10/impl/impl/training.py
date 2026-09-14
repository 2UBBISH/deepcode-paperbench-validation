'''Adam training loop with validation split and early stopping.

This module trains conditional score networks with the denoising score
matching objectives from :mod:`impl.losses`.  It holds back a validation split,
uses Adam, and returns the checkpoint with the lowest validation loss.
'''

from __future__ import annotations

import copy

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

from impl.config import (
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    MAX_TRAINING_ITERATIONS,
    VALIDATION_FRACTION,
    TrialConfig,
)
from impl.losses import (
    denoising_likelihood_score_matching_loss,
    denoising_posterior_score_matching_loss,
)


class _PairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = [(torch.as_tensor(a), torch.as_tensor(b)) for a, b in pairs]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def _nested_pair_to_dataset(dataset):
    if isinstance(dataset, Dataset):
        return dataset
    if isinstance(dataset, (tuple, list)) and len(dataset) == 2:
        theta, x = dataset
        return TensorDataset(torch.as_tensor(theta), torch.as_tensor(x))
    try:
        pairs = [(torch.as_tensor(a), torch.as_tensor(b)) for a, b in dataset]
    except Exception as exc:
        raise TypeError(
            'dataset must be a torch Dataset or an iterable of (theta, x) pairs'
        ) from exc
    return _PairDataset(pairs)


def _param(params, key, default):
    if isinstance(params, TrialConfig):
        return getattr(params, key, default)
    if isinstance(params, dict):
        return params.get(key, default)
    return default


class AdamTrainer:
    '''Adam trainer with validation early stopping.

    Parameters
    ----------
    model:
        Conditional score network to train.
    lr:
        Adam learning rate.
    patience:
        Number of validation evaluations without improvement before stopping.
    max_iters:
        Maximum number of training epochs.
    '''

    def __init__(
        self,
        model,
        lr: float = LEARNING_RATE,
        patience: int = EARLY_STOPPING_PATIENCE,
        max_iters: int = MAX_TRAINING_ITERATIONS,
    ):
        self.model = model
        self.lr = float(lr)
        self.patience = int(patience)
        self.max_iters = int(max_iters)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

    def _batch_to_tensors(self, batch, device):
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            theta_0, x = batch[0], batch[1]
        else:
            theta_0, x = batch
        return theta_0.to(device), x.to(device)

    def _call_loss(self, loss_fn, score, theta_t, theta_0, sde, t, loss_kwargs):
        if loss_fn is denoising_likelihood_score_matching_loss:
            if not loss_kwargs or 'prior_score' not in loss_kwargs:
                raise ValueError(
                    'denoising_likelihood_score_matching_loss requires loss_kwargs '
                    "with key 'prior_score'"
                )
            prior_score = loss_kwargs['prior_score']
            return loss_fn(score, theta_t, theta_0, prior_score, sde, t=t)
        return loss_fn(score, theta_t, theta_0, sde, t=t)

    def _evaluate(self, loader, sde, loss_fn, loss_kwargs, device):
        self.model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for batch in loader:
                theta_0, x = self._batch_to_tensors(batch, device)
                n = theta_0.size(0)
                t = torch.rand(n, 1, device=device) * (sde.t_max - sde.t_min) + sde.t_min
                noise = torch.randn_like(theta_0)
                theta_t = sde.sample_transition(theta_0, t, noise)
                score = self.model(theta_t, x, t)
                loss = self._call_loss(
                    loss_fn, score, theta_t, theta_0, sde, t, loss_kwargs
                )
                total += float(loss.detach().item()) * n
                count += n
        if count == 0:
            return float('inf')
        return total / count

    def fit(
        self,
        train_loader,
        valid_loader,
        sde,
        loss_fn=denoising_posterior_score_matching_loss,
        loss_kwargs=None,
        device='cpu',
    ):
        model = self.model
        model.to(device)
        best_loss = float('inf')
        best_state = copy.deepcopy(model.state_dict())
        steps_no_improve = 0

        for _ in range(self.max_iters):
            model.train()
            for batch in train_loader:
                theta_0, x = self._batch_to_tensors(batch, device)
                n = theta_0.size(0)
                t = torch.rand(n, 1, device=device) * (sde.t_max - sde.t_min) + sde.t_min
                noise = torch.randn_like(theta_0)
                theta_t = sde.sample_transition(theta_0, t, noise)
                score = model(theta_t, x, t)
                loss = self._call_loss(
                    loss_fn, score, theta_t, theta_0, sde, t, loss_kwargs
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            eval_loader = valid_loader
            if eval_loader is None or len(eval_loader.dataset) == 0:
                eval_loader = train_loader
            valid_loss = self._evaluate(
                eval_loader, sde, loss_fn, loss_kwargs, device
            )

            if valid_loss < best_loss - 1e-12:
                best_loss = valid_loss
                best_state = copy.deepcopy(model.state_dict())
                steps_no_improve = 0
            else:
                steps_no_improve += 1
            if steps_no_improve >= self.patience:
                break

        model.load_state_dict(best_state)
        return model


def train_score_network(model, dataset, params, sde=None, loss_fn=None, loss_kwargs=None):
    '''Train a conditional score network and return the best checkpoint.

    ``params`` may be a :class:`TrialConfig` instance or a dictionary with keys
    ``training_steps``, ``batch_size``, ``validation_fraction``,
    ``early_stopping_patience``, ``learning_rate``, ``seed``, and ``device``.
    If ``sde`` is not supplied, it is read from ``params.sde`` or
    ``params['sde']``.
    '''
    if sde is None:
        sde = _param(params, 'sde', None)
    if sde is None:
        raise ValueError(
            'train_score_network requires an SDE. Pass it as the sde argument or '
            "store it in params as 'sde'."
        )

    ds = _nested_pair_to_dataset(dataset)
    if len(ds) == 0:
        raise ValueError('cannot train on an empty dataset')

    batch_size = max(1, int(_param(params, 'batch_size', 50)))
    validation_fraction = float(
        _param(params, 'validation_fraction', VALIDATION_FRACTION)
    )
    validation_fraction = min(max(validation_fraction, 0.0), 0.5)
    patience = int(_param(params, 'early_stopping_patience', EARLY_STOPPING_PATIENCE))
    max_iters = int(_param(params, 'training_steps', MAX_TRAINING_ITERATIONS))
    lr = float(_param(params, 'learning_rate', LEARNING_RATE))
    seed = int(_param(params, 'seed', 0))
    device = str(_param(params, 'device', 'cpu'))

    n_total = len(ds)
    if n_total >= 2:
        valid_size = max(1, int(n_total * validation_fraction))
        train_size = n_total - valid_size
        generator = torch.Generator().manual_seed(seed)
        train_ds, valid_ds = random_split(
            ds, [train_size, valid_size], generator=generator
        )
    else:
        train_ds = ds
        valid_ds = None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = None
    if valid_ds is not None:
        valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    chosen_loss = loss_fn or denoising_posterior_score_matching_loss
    trainer = AdamTrainer(model, lr=lr, patience=patience, max_iters=max_iters)
    return trainer.fit(
        train_loader,
        valid_loader,
        sde,
        loss_fn=chosen_loss,
        loss_kwargs=loss_kwargs,
        device=device,
    )
