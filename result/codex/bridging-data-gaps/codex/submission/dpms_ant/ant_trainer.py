"""Algorithm 1: training DPMs with ANT (adversarial noise transfer).

Pseudo-code of the paper:

    Require: binary classifier p_phi, pre-trained DPMs eps_theta, learning rate eta
    repeat
        x_0 ~ q(x_0)
        t ~ Uniform({1, ..., T})
        eps ~ N(0, I)
        for j = 0, ..., J-1 do
            Update eps^j via Equation (7)
        end for
        Compute L(psi) with eps* = eps^J via Equation (8)
        Update the adaptor model parameter: psi = psi - eta grad_psi L(psi)
    until converged

`ANTTrainer` implements exactly this.  Flags switch off the two contributions
so that every row of the ablation study (Figure 4 / Figure 6) can be produced:

* ``use_similarity_guidance=False, use_adversarial_noise=False``  -> baseline
  (traditional DDPM fine-tuning; Eq. 1)
* ``use_adversarial_noise=False``                                 -> DPMs-ANT w/o AN
* ``use_similarity_guidance=False``                               -> AN only
* both ``True``                                                   -> DPMs-ANT

The same trainer is used for the DDPM and the LDM backbone (the backbone is
just an ``eps_theta``); ``train_full_finetune`` provides the "direct fine-tuning
of the whole model" baseline of Figure 4.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

from .adaptor import AdaptorConfig, adaptor_parameters, set_adaptor_training
from .adversarial_noise import AdversarialNoiseConfig, select_adversarial_noise
from .guidance import classifier_guidance, similarity_guided_loss
from .schedules import DiffusionSchedule
from .utils import BatchIterator, Logger, ensure_dir, progbar


@dataclass
class ANTConfig:
    """Training configuration.

    Defaults follow Section 5.2 / the supplementary material for DDPM:
    ``lr = 5e-5``, ``~300`` iterations, batch size ``40``, ``gamma = 5``,
    ``J = 10``, ``omega = 0.02``.  The per-task values of Table 3 are in
    :mod:`dpms_ant.configs`.
    """

    iterations: int = 300
    batch_size: int = 40
    lr: float = 5e-5
    betas: tuple = (0.9, 0.999)
    gamma: float = 5.0
    use_similarity_guidance: bool = True
    use_adversarial_noise: bool = True
    adaptor: AdaptorConfig = field(default_factory=AdaptorConfig)
    adversarial: AdversarialNoiseConfig = field(default_factory=AdversarialNoiseConfig)
    log_every: int = 25
    seed: int = 0
    save_every: int = 0
    save_dir: Optional[str] = None


def eps_predictor(model: nn.Module, in_channels: int = 3) -> Callable:
    """Turn a denoiser into a pure epsilon predictor (handles ``learn_sigma``)."""

    def predict(x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        out = model(x, timesteps)
        if isinstance(out, tuple):
            out = out[0]
        if out.shape[1] == 2 * in_channels:
            out = torch.split(out, in_channels, dim=1)[0]
        return out

    return predict


class ANTTrainer:
    """Adapter-only transfer learning with similarity guidance + AN selection."""

    def __init__(
        self,
        model: nn.Module,
        classifier: Optional[nn.Module],
        schedule: DiffusionSchedule,
        config: Optional[ANTConfig] = None,
        device: str = "cpu",
        logger: Optional[Logger] = None,
        in_channels: int = 3,
    ):
        self.config = config or ANTConfig()
        self.model = model.to(device)
        self.classifier = None if classifier is None else classifier.to(device)
        self.schedule = schedule.to(device)
        self.device = torch.device(device)
        self.logger = logger
        self.in_channels = in_channels
        self.predict_eps = eps_predictor(self.model, in_channels=in_channels)
        self.history: Dict[str, List[float]] = {"loss": [], "grad_norm": []}

    # ------------------------------------------------------------------ #
    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger.log(message)
        else:
            print(message, flush=True)

    def _sample_timesteps(self, batch_size: int, generator: torch.Generator) -> torch.Tensor:
        # Algorithm 1: t ~ Uniform({1, ..., T})
        return torch.randint(
            1, self.schedule.num_timesteps, (batch_size,), device=self.device, generator=generator
        ).long()

    def build_optimizer(self, parameters) -> torch.optim.Optimizer:
        return torch.optim.Adam(parameters, lr=self.config.lr, betas=self.config.betas)

    # ------------------------------------------------------------------ #
    def train_step(self, x_start: torch.Tensor, generator: torch.Generator) -> Dict[str, float]:
        """One iteration of Algorithm 1."""
        config = self.config
        batch_size = x_start.shape[0]
        timesteps = self._sample_timesteps(batch_size, generator)
        noise = torch.randn(
            x_start.shape, device=self.device, dtype=x_start.dtype, generator=generator
        )

        # ---- inner maximisation: adversarial noise selection (Eq. 7) ----
        if config.use_adversarial_noise:
            eps_star = select_adversarial_noise(
                model=self.model,
                schedule=self.schedule,
                x_start=x_start,
                timesteps=timesteps,
                config=config.adversarial,
                initial_noise=noise,
            )
        else:
            eps_star = noise

        # the corresponding noised image x_t^* = sqrt(alpha_bar) x0 + sqrt(1-alpha_bar) eps^*
        x_t_star = self.schedule.q_sample(x_start, timesteps, eps_star)

        # ---- similarity guidance from the frozen classifier (Eq. 5) ----
        guidance = None
        if config.use_similarity_guidance:
            if self.classifier is None:
                raise ValueError("similarity-guided training requires a classifier")
            guidance = classifier_guidance(
                classifier=self.classifier,
                schedule=self.schedule,
                x_t=x_t_star,
                timesteps=timesteps,
                target_class=1,  # 0 = source, 1 = target
                gamma=config.gamma,
            )

        # ---- L(psi) (Eq. 8) ----
        eps_pred = self.predict_eps(x_t_star, timesteps)
        loss = similarity_guided_loss(eps_pred, eps_star, guidance=guidance)
        return {"loss": loss, "eps_pred": eps_pred, "guidance": guidance}

    # ------------------------------------------------------------------ #
    def train(self, batches: Callable[[], torch.Tensor]) -> Dict[str, List[float]]:
        """Run Algorithm 1 for ``config.iterations`` iterations."""
        config = self.config
        torch.manual_seed(config.seed)
        generator = torch.Generator(device=self.device).manual_seed(config.seed)

        parameters = adaptor_parameters(self.model)
        if not parameters:
            raise RuntimeError(
                "no adaptor parameters found -- call dpms_ant.add_adaptors(model, ...) first"
            )
        optimizer = self.build_optimizer(parameters)
        batches = BatchIterator(batches)
        self.model.to(self.device)
        if self.classifier is not None:
            self.classifier.eval()
            for parameter in self.classifier.parameters():
                parameter.requires_grad_(False)

        start = time.time()
        for step in progbar(range(config.iterations), desc="ANT"):
            set_adaptor_training(self.model, True)
            x_start = next(batches)
            x_start = x_start.to(device=self.device, dtype=torch.float32)
            outputs = self.train_step(x_start, generator)
            loss = outputs["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=1e9)
            optimizer.step()

            self.history["loss"].append(float(loss.detach()))
            self.history["grad_norm"].append(float(grad_norm))
            if step % config.log_every == 0 or step == config.iterations - 1:
                self._log(
                    f"[ANT] iter {step:5d}/{config.iterations} loss {float(loss):.4f} "
                    f"grad {float(grad_norm):.3e} ({(time.time() - start):.1f}s)"
                )
            if config.save_every and config.save_dir and (step + 1) % config.save_every == 0:
                self.save_checkpoint(step + 1)
        self.history["time"] = [time.time() - start]
        return self.history

    # ------------------------------------------------------------------ #
    def save_checkpoint(self, step: Optional[int] = None) -> str:
        assert self.config.save_dir is not None
        ensure_dir(self.config.save_dir)
        name = "adaptor.pt" if step is None else f"adaptor_{step:06d}.pt"
        path = f"{self.config.save_dir.rstrip('/')}/{name}"
        payload = {
            "step": step,
            "adaptor": {
                name: parameter.detach().cpu()
                for name, parameter in self.model.named_parameters()
                if ".adaptor." in name
            },
            "config": self.config,
        }
        torch.save(payload, path)
        return path


def train_full_finetune(
    model: nn.Module,
    classifier: Optional[nn.Module],
    schedule: DiffusionSchedule,
    batches: Callable[[], torch.Tensor],
    config: Optional[ANTConfig] = None,
    device: str = "cpu",
    logger: Optional[Logger] = None,
    in_channels: int = 3,
) -> Dict[str, List[float]]:
    """The "direct fine-tuning of the whole model" baseline of Figure 4.

    Identical to Algorithm 1 but every pre-trained parameter is trainable.
    """
    config = config or copy.deepcopy(ANTConfig())
    model = model.to(device)
    schedule = schedule.to(device)
    predict_eps = eps_predictor(model, in_channels=in_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=config.betas)
    generator = torch.Generator(device=device).manual_seed(config.seed)
    history: Dict[str, List[float]] = {"loss": []}
    batches = BatchIterator(batches)

    for step in progbar(range(config.iterations), desc="fine-tune"):
        model.train()
        x_start = next(batches)
        x_start = x_start.to(device=device, dtype=torch.float32)
        batch_size = x_start.shape[0]
        timesteps = torch.randint(
            1, schedule.num_timesteps, (batch_size,), device=device, generator=generator
        ).long()
        noise = torch.randn(x_start.shape, device=device, generator=generator)
        if config.use_adversarial_noise:
            eps_star = select_adversarial_noise(
                model, schedule, x_start, timesteps, config=config.adversarial, initial_noise=noise
            )
        else:
            eps_star = noise
        x_t = schedule.q_sample(x_start, timesteps, eps_star)
        guidance = None
        if config.use_similarity_guidance and classifier is not None:
            guidance = classifier_guidance(
                classifier, schedule, x_t, timesteps, target_class=1, gamma=config.gamma
            )
        eps_pred = predict_eps(x_t, timesteps)
        loss = similarity_guided_loss(eps_pred, eps_star, guidance=guidance)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history["loss"].append(float(loss.detach()))
        if logger is not None and (step % config.log_every == 0 or step == config.iterations - 1):
            logger.log(f"[fine-tune] iter {step:5d} loss {float(loss):.4f}")
    return history
